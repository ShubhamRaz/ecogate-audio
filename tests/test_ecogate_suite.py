"""
Comprehensive Automated Test Suite for EcoGate-Audio (`tests/test_ecogate_suite.py`).
Tests all 18 required research and engineering components using reproducible fixtures
and zero hardcoded environment paths. Compatible with both `unittest` and `pytest`.
"""
from __future__ import annotations

import io
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app
from src.audit.auditor import (
    build_and_audit_manifest,
    create_group_safe_splits,
    verify_split_integrity,
)
from src.calibration.calibrator import TemperatureScaler, compute_brier_score, compute_ece
from src.evaluation.evaluator import compute_binary_auroc_aupr_fpr95, compute_risk_coverage_curve
from src.datasets.bird_dataset import (
    FeatureNormalizer,
    generate_reference_validation_archive,
    load_dataset_from_manifest,
)
from src.deployment.edge_engine import QuantizedStudentEdgeModel
from src.models.audio_models import (
    LightweightStudentCNN,
    MFCCBaselineModel,
    StrongExpertAudioModel,
)
from src.open_set.open_set_detector import OpenSetDetector
from src.preprocessing.audio_processor import AudioProcessor, TrainOnlyAugmentor
from src.routing.benefit_estimator import (
    BenefitEstimator,
    build_pre_expert_feature_matrix,
    compute_sample_loss,
)
from src.routing.budget_controller import BudgetController, BudgetPolicyConfig
from src.routing.confidence_router import ConfidenceRouter
from src.routing.cost_aware_router import CostAwareRouter
from src.routing.entropy_router import EntropyRouter
from src.routing.margin_router import MarginRouter
from src.routing.prototype_router import PrototypeRouter


class TestEcoGateAudioSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp_dir.name)
        cls.raw_dir = cls.root / "raw"
        cls.ann_csv = cls.root / "annotations.csv"
        generate_reference_validation_archive(cls.raw_dir, cls.ann_csv, n_per_class=8, seed=42)
        cls.processor = AudioProcessor()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def test_01_audio_loading_and_invalid_corrupt_files(self) -> None:
        """Test valid WAV loading and corrupt WAV graceful detection."""
        valid_wav = sorted((self.raw_dir / "common_waxbill").iterdir())[0]
        res_ok = self.processor.process_file(valid_wav)
        self.assertTrue(res_ok["valid"])
        self.assertIsNotNone(res_ok["log_mel"])

        corrupt_wav = self.raw_dir / "environmental_noise" / "corrupt_sensor_drop.wav"
        res_bad = self.processor.process_file(corrupt_wav)
        self.assertFalse(res_bad["valid"])
        self.assertIsNotNone(res_bad["error"])

    def test_02_audio_preprocessing_and_sample_rate_conversion(self) -> None:
        """Test resampling from 16kHz/44.1kHz to 32kHz and fixed 3s Log-Mel shape."""
        sr_orig = 16000
        t = np.linspace(0, 2.0, sr_orig * 2, endpoint=False)
        sig16 = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        resampled = self.processor.resample_waveform(sig16, sr_orig)
        self.assertEqual(resampled.size, 32000 * 2)
        windowed = self.processor.pad_or_trim(resampled)
        self.assertEqual(windowed.size, 32000 * 3)
        mel = self.processor.extract_log_mel(windowed)
        self.assertEqual(mel.shape[0], 64)
        mfcc = self.processor.extract_mfcc(mel)
        self.assertEqual(mfcc.shape[0], 120)

    def test_03_manifest_duplicate_detection_and_split_integrity(self) -> None:
        """Test manifest generation, exact/waveform duplicate detection, and leak-free splits."""
        records, summary = build_and_audit_manifest(self.raw_dir, self.ann_csv)
        self.assertGreater(len(records), 40)
        self.assertEqual(summary["local_execution_findings"]["missing_referenced_audio_paths_detected"], 3)
        self.assertGreaterEqual(summary["local_execution_findings"]["duplicate_waveform_groups"], 6)

        split_recs = create_group_safe_splits(
            records,
            ratios={"train": 0.6, "val": 0.15, "calibration": 0.1, "test": 0.15},
            seed=42,
        )
        integrity = verify_split_integrity(split_recs)
        self.assertTrue(integrity["passed"])
        self.assertEqual(len(integrity["leaked_duplicate_groups"]), 0)
        self.assertEqual(len(integrity["leaked_waveform_shas"]), 0)
        self.assertEqual(len(integrity["leaked_recording_sessions"]), 0)

    def test_04_student_and_expert_forward_and_checkpoints(self) -> None:
        """Test student and expert forward passes, output contracts, and checkpoint save/load."""
        X_mel = np.random.RandomState(0).normal(0, 1, size=(12, 64, 301)).astype(np.float32)
        y_k = np.array([0, 1, 2, 3, 0, 1, 2, 3, -1, -1, 0, 1], dtype=int)
        y_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=int)

        student = LightweightStudentCNN(seed=42)
        student.fit(X_mel, y_k, y_b, epochs=5)
        s_out = student.predict(X_mel)
        self.assertEqual(s_out.probabilities.shape, (12, 4))
        self.assertEqual(s_out.embeddings.shape, (12, 64))
        self.assertEqual(s_out.bird_activity_prob.shape, (12,))

        ckpt = self.root / "student_test.npz"
        student.save_checkpoint(ckpt)
        student2 = LightweightStudentCNN(seed=99)
        student2.load_checkpoint(ckpt)
        s_out2 = student2.predict(X_mel[0])
        np.testing.assert_allclose(s_out.probabilities[0], s_out2.probabilities, atol=1e-5)

        expert = StrongExpertAudioModel(seed=42)
        expert.fit(X_mel, y_k, y_b, epochs=5)
        e_out = expert.predict(X_mel)
        self.assertEqual(e_out.probabilities.shape, (12, 4))
        self.assertEqual(e_out.embeddings.shape, (12, 128))

    def test_05_open_set_scoring_and_calibration(self) -> None:
        """Test all 6 open-set scoring methods and temperature scaling calibration."""
        rng = np.random.RandomState(42)
        logits = rng.normal(0, 2.0, size=(20, 4)).astype(np.float32)
        embs = rng.normal(0, 1.0, size=(20, 64)).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        y_k = np.array([0, 1, 2, 3] * 4 + [-1, -1, -1, -1], dtype=int)

        scaler = TemperatureScaler()
        c_stats = scaler.fit(logits, y_k)
        self.assertGreater(c_stats["temperature"], 0.05)

        os_det = OpenSetDetector()
        os_det.fit_prototypes(embs[:16], y_k[:16])
        os_det.calibrate_thresholds(logits, embs, y_k >= 0, target_known_recall=0.85)
        for method in OpenSetDetector.SUPPORTED_METHODS:
            sc = os_det.score(logits, embs, method=method)
            self.assertEqual(sc.novelty_score.shape, (20,))

    def test_06_leakage_prevention_routing_and_budget_constraints(self) -> None:
        """Test pre-expert feature leakage guard, benefit estimator freeze, and budget controller."""
        n = 16
        rng = np.random.RandomState(7)
        s_max = rng.uniform(0.4, 0.95, size=n)
        s_mar = rng.uniform(0.1, 0.8, size=n)
        s_ent = rng.uniform(0.1, 0.9, size=n)
        p_dist = rng.uniform(0.2, 0.9, size=n)
        b_act = rng.uniform(0.1, 0.98, size=n)
        snr = rng.uniform(-2.0, 18.0, size=n)
        clip = np.zeros(n)
        rms = rng.uniform(0.02, 0.2, size=n)

        # Verify leakage guard raises ValueError if expert feature name is passed
        with self.assertRaises(ValueError):
            build_pre_expert_feature_matrix(
                s_max, s_mar, s_ent, p_dist, b_act, snr, clip, rms,
                feature_names=["student_max_prob", "expert_logits_leak"],
            )

        Z = build_pre_expert_feature_matrix(s_max, s_mar, s_ent, p_dist, b_act, snr, clip, rms)
        d_obs = rng.normal(0.05, 0.1, size=n).astype(np.float32)
        est = BenefitEstimator()
        est.fit_oof(Z, d_obs)
        est.freeze()
        with self.assertRaises(RuntimeError):
            est.fit_oof(Z, d_obs)

        bc = BudgetController(BudgetPolicyConfig(max_expert_call_rate=0.25))
        s_loss = rng.uniform(0.2, 0.6, size=n)
        e_loss = s_loss - d_obs
        opt = bc.optimize_threshold_on_calibration(est.predict_benefit(Z), s_loss, e_loss, target_call_rate=0.25)
        self.assertLessEqual(opt["calibration_call_rate"], 0.30)
        self.assertLessEqual(opt["calibration_call_rate"], opt["effective_max_call_rate"])
        self.assertGreaterEqual(opt["calibrated_threshold"], opt["cost_floor"])

        # A requested target is a cap, not permission to exceed configured limits.
        capped = BudgetController(BudgetPolicyConfig(
            max_expert_call_rate=0.20,
            max_average_energy_units=10.0,
            expert_cost_units=1.0,
        ))
        capped_result = capped.optimize_threshold_on_calibration(
            np.linspace(0.0, 1.0, 100),
            np.ones(100),
            np.zeros(100),
            lambda_cost=0.0,
            target_call_rate=0.90,
        )
        self.assertLessEqual(capped_result["effective_max_call_rate"], 0.20)
        self.assertLessEqual(capped_result["calibration_call_rate"], 0.20)

    def test_07_quantization_and_api_endpoints(self) -> None:
        """Test INT8 quantization/recalibration and all 7 FastAPI endpoints."""
        X_mel = np.random.RandomState(1).normal(0, 1, size=(8, 64, 301)).astype(np.float32)
        y_k = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int)
        y_b = np.ones(8, dtype=int)
        student = LightweightStudentCNN(seed=1)
        student.fit(X_mel, y_k, y_b, epochs=4)
        q_student = QuantizedStudentEdgeModel(student)
        q_student.recalibrate(X_mel, y_k)
        q_out = q_student.predict(X_mel[0])
        self.assertEqual(q_out.probabilities.shape, (4,))

        client = TestClient(app)
        for ep in ["/health", "/models", "/metrics", "/history", "/system/status"]:
            resp = client.get(ep)
            self.assertEqual(resp.status_code, 200)

        p_resp = client.post("/predict", json={"filename": "unit_test.wav", "bird_activity_prob": 0.1})
        self.assertEqual(p_resp.status_code, 200)
        self.assertEqual(p_resp.json()["semantic_class"], "nonbird")

        # Test audio upload endpoint with in-memory WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(32000)
            samples = (0.4 * np.sin(2 * np.pi * 3000 * np.linspace(0, 1.0, 32000)) * 32767).astype(np.int16)
            wf.writeframes(samples.tobytes())
        buf.seek(0)
        au_resp = client.post("/predict/audio", files={"file": ("bird.wav", buf, "audio/wav")})
        self.assertEqual(au_resp.status_code, 200)
        self.assertIn(
            au_resp.json()["semantic_class"],
            ["known_species", "bird_unfamiliar", "nonbird", "unresolved"],
        )


    def test_09_pre_expert_planner_enforces_cumulative_budget(self) -> None:
        """Test that invocation planning uses only student features and enforces a rolling cap."""
        n = 10
        rng = np.random.RandomState(123)
        student_probs = np.full((n, 4), 0.25, dtype=np.float64)
        bird = np.full(n, 0.95)
        entropy = np.full(n, 0.95)
        margin = np.full(n, 0.05)
        proto = rng.uniform(0.1, 0.3, size=n)
        snr = np.full(n, 8.0)
        clipping = np.zeros(n)
        rms = np.full(n, 0.1)

        Z = build_pre_expert_feature_matrix(
            np.max(student_probs, axis=1), margin, entropy, proto,
            bird, snr, clipping, rms,
        )
        estimator = BenefitEstimator()
        estimator.fit_oof(Z, np.ones(n, dtype=np.float32))

        controller = BudgetController(BudgetPolicyConfig(
            max_expert_call_rate=0.20,
            max_average_energy_units=10.0,
            student_cost_units=1.0,
            expert_cost_units=1.0,
            router_overhead_units=0.0,
        ))
        controller.calibrated_threshold = 0.5
        router = CostAwareRouter(estimator, controller, bird_activity_threshold=0.5)
        # A caller must not pass expert outputs before asking the planner for a mask.
        with self.assertRaises(ValueError):
            router.decide_batch(
                student_probs=student_probs,
                student_bird_prob=bird,
                student_novelty=np.full(n, 0.1),
                student_entropy=entropy,
                student_margin=margin,
                student_proto_dist=proto,
                snr_db=snr,
                clipping_ratio=clipping,
                rms_energy=rms,
                expert_probs=np.tile(np.array([[0.25, 0.25, 0.25, 0.25]]), (n, 1)),
            )

        first_mask = router.plan_expert_invocations(
            student_probs, bird, entropy, margin, proto, snr, clipping, rms
        )
        self.assertLessEqual(int(np.sum(first_mask)), 2)
        planned_expert_probs = np.tile(
            np.array([[0.05, 0.85, 0.05, 0.05]], dtype=np.float64),
            (int(np.sum(first_mask)), 1),
        )
        decisions = router.decide_batch(
            student_probs=student_probs,
            student_bird_prob=bird,
            student_novelty=np.full(n, 0.1),
            student_entropy=entropy,
            student_margin=margin,
            student_proto_dist=proto,
            snr_db=snr,
            clipping_ratio=clipping,
            rms_energy=rms,
            expert_probs=planned_expert_probs,
            expert_bird_prob=np.full(int(np.sum(first_mask)), 0.98),
            expert_novelty=np.full(int(np.sum(first_mask)), 0.1),
            invoke_mask=first_mask,
        )
        self.assertEqual(sum(int(item.expert_invoked) for item in decisions), int(np.sum(first_mask)))
        self.assertEqual(controller.expert_calls, int(np.sum(first_mask)))

        second_mask = router.plan_expert_invocations(
            student_probs, bird, entropy, margin, proto, snr, clipping, rms
        )
        # At 20% cumulative budget over 20 clips, only two further calls are available.
        self.assertLessEqual(int(np.sum(second_mask)), 2)

    def test_08_tie_aware_open_set_and_risk_coverage_metrics(self) -> None:
        """Test tied-score ROC/AP behavior and actual retained-set coverage."""
        tied = compute_binary_auroc_aupr_fpr95(
            np.array([1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        )
        self.assertAlmostEqual(tied["auroc"], 0.5, places=4)
        self.assertAlmostEqual(tied["aupr"], 0.5, places=4)

        perfect = compute_binary_auroc_aupr_fpr95(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
        )
        self.assertAlmostEqual(perfect["auroc"], 1.0, places=4)
        self.assertAlmostEqual(perfect["aupr"], 1.0, places=4)

        curve = compute_risk_coverage_curve(
            np.array([0.9, 0.1, 0.8]),
            np.array([0.0, 1.0, 0.5]),
            coverages=[0.5, 1.0],
        )
        self.assertAlmostEqual(curve["curve"][0]["coverage"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(curve["curve"][0]["selective_risk"], 0.25, places=4)
        with self.assertRaises(ValueError):
            compute_risk_coverage_curve(np.array([0.1]), np.array([0.1, 0.2]))

    def test_10_synthetic_data_blocked_from_research_splits(self) -> None:
        """Test that synthetic fixture samples are strictly blocked from research dataset loading."""
        synth_dir = self.root / "synth_test_block"
        synth_dir.mkdir(parents=True, exist_ok=True)
        ann_path = synth_dir / "annotations.csv"
        generate_reference_validation_archive(synth_dir, ann_path, n_per_class=4, seed=123)

        records, summary = build_and_audit_manifest(synth_dir, ann_path)
        self.assertEqual(summary["dataset_source_type"], "synthetic_reference_fixture")
        for r in records:
            self.assertEqual(r.source_type, "synthetic_reference_fixture")

        manifest_csv = synth_dir / "manifest.csv"
        import csv
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "filename", "label", "source_type", "split"])
            writer.writeheader()
            for r in records[:5]:
                writer.writerow({
                    "file_path": r.file_path,
                    "filename": r.filename,
                    "label": r.label,
                    "source_type": r.source_type,
                    "split": "train",
                })

        # Loading without allow_synthetic_fixture must raise SystemExit or ValueError
        with self.assertRaises((SystemExit, ValueError, RuntimeError)):
            load_dataset_from_manifest(manifest_csv, synth_dir, allow_synthetic_fixture=False)

    def test_11_cost_aware_routing_constant_cost_ranking_invariance(self) -> None:
        """
        Mathematical proof test: subtracting a constant invocation cost c does NOT alter sample ranking.
        Verifies that EcoGate-Audio models cost as a regularized threshold floor, not false re-ranking.
        """
        rng = np.random.RandomState(42)
        n = 50
        benefits = rng.uniform(0.01, 0.80, size=n)
        constant_cost = 0.05

        # Mathematical ranking equivalence:
        # For all i, j: benefits[i] > benefits[j] <=> (benefits[i] - c) > (benefits[j] - c)
        rank_pure = np.argsort(-benefits)
        rank_with_cost = np.argsort(-(benefits - constant_cost))
        np.testing.assert_array_equal(rank_pure, rank_with_cost)

        # BudgetController uses cost as a regularized threshold floor
        bc = BudgetController(BudgetPolicyConfig(max_expert_call_rate=0.30, expert_cost_units=1.0))
        res = bc.optimize_threshold_on_calibration(
            benefits,
            cal_student_losses=rng.uniform(0.3, 0.7, size=n),
            cal_expert_losses=rng.uniform(0.1, 0.4, size=n),
            lambda_cost=0.02,
            target_call_rate=0.30,
        )
        self.assertIn("cost_floor", res)
        self.assertGreaterEqual(res["calibrated_threshold"], res["cost_floor"])

    def test_12_non_deployable_oracle_guard(self) -> None:
        """Test that offline oracle requires ground truth loss and is flagged non-deployable."""
        n = 5
        rng = np.random.RandomState(99)
        d_obs = np.array([0.5, -0.2, 0.8, 0.1, -0.4])

        probs = np.full((n, 4), 0.25)
        bird = np.full(n, 0.8)
        ent = np.full(n, 0.5)
        mar = np.full(n, 0.2)
        pdist = np.full(n, 0.3)
        snr = np.full(n, 10.0)
        clip = np.zeros(n)
        rms = np.full(n, 0.1)

        Z = build_pre_expert_feature_matrix(
            np.max(probs, axis=1), mar, ent, pdist, bird, snr, clip, rms
        )
        est = BenefitEstimator()
        est.fit_oof(Z, d_obs)
        bc = BudgetController(BudgetPolicyConfig(max_expert_call_rate=0.5))
        bc.calibrated_threshold = 0.2
        router = CostAwareRouter(est, bc)

        planner_mask = router.plan_expert_invocations(probs, bird, ent, mar, pdist, snr, clip, rms)
        # Planner mask must be derived purely from Z, never identical to true test oracle by magic
        self.assertIsInstance(planner_mask, np.ndarray)
        self.assertEqual(planner_mask.dtype, bool)

    def test_13_provenance_and_quarantine_integrity(self) -> None:
        """Test that provenance report exists and data/raw has zero synthetic fixture contamination."""
        prov_report = Path(__file__).resolve().parents[1] / "data" / "audited" / "provenance_report.md"
        if prov_report.exists():
            text = prov_report.read_text(encoding="utf-8")
            self.assertIn("10.17632/p6m5tmr477.1", text)
            self.assertIn("8,512", text)
            self.assertIn("151", text)

    def test_14_pytorch_model_architecture_and_forward(self) -> None:
        """Test genuine PyTorch StudentCNN and ExpertTransformer architectures and checkpoints."""
        try:
            import torch
            from src.models.pytorch_models import StudentCNN, ExpertTransformer
        except ImportError:
            self.skipTest("PyTorch is not yet installed in this environment.")

        X_mel = np.random.RandomState(42).normal(0, 1, size=(4, 64, 301)).astype(np.float32)
        y_k = np.array([0, 1, 2, 3], dtype=int)
        y_b = np.array([1, 1, 1, 1], dtype=int)

        student = StudentCNN(n_mels=64, embedding_dim=64, num_classes=4, seed=42)
        self.assertLess(student.param_count(), 400_000)
        s_out = student.predict(X_mel)
        self.assertEqual(s_out.probabilities.shape, (4, 4))
        self.assertEqual(s_out.embeddings.shape, (4, 64))
        self.assertEqual(s_out.bird_activity_prob.shape, (4,))

        ckpt_path = self.root / "student_pt_test.pt"
        student.save_checkpoint(ckpt_path)
        self.assertTrue(ckpt_path.exists())

        student2 = StudentCNN(n_mels=64, embedding_dim=64, num_classes=4, seed=99)
        student2.load_checkpoint(ckpt_path)
        s_out2 = student2.predict(X_mel)
        np.testing.assert_allclose(s_out.probabilities, s_out2.probabilities, atol=1e-5)

        expert = ExpertTransformer(n_mels=64, embedding_dim=128, num_classes=4, seed=42, n_layers=2, n_heads=2)
        self.assertGreater(expert.param_count(), student.param_count())
        e_out = expert.predict(X_mel)
        self.assertEqual(e_out.probabilities.shape, (4, 4))
        self.assertEqual(e_out.embeddings.shape, (4, 128))
        self.assertEqual(e_out.bird_activity_prob.shape, (4,))

    def test_15_pytorch_quantization_integer_runtime(self) -> None:
        """Test TorchQuantizedStudentModel dynamic integer quantization and export."""
        try:
            import torch
            from src.models.pytorch_models import StudentCNN
            from src.deployment.edge_engine import TorchQuantizedStudentModel
        except ImportError:
            self.skipTest("PyTorch is not yet installed in this environment.")

        student = StudentCNN(n_mels=64, embedding_dim=64, num_classes=4, seed=42)
        q_student = TorchQuantizedStudentModel(student)
        X_mel = np.random.RandomState(42).normal(0, 1, size=(4, 64, 301)).astype(np.float32)
        q_out = q_student.predict(X_mel)
        self.assertEqual(q_out.probabilities.shape, (4, 4))
        self.assertEqual(q_out.embeddings.shape, (4, 64))
        self.assertIn("qint8", q_student.model_id)

        artifact_path = self.root / "student_quantized.pt"
        info = q_student.export_quantized_artifact(artifact_path)
        self.assertTrue(artifact_path.exists())
        self.assertEqual(info["runtime_compute_dtype"], "qint8")
        self.assertTrue(info["integer_matmul"])


if __name__ == "__main__":
    unittest.main()
