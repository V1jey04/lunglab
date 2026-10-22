import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Lung CT radiomics research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download public Lung1 features and official labels")
    download.add_argument("--raw", default="data/raw/nsclc_radiomics")
    download.add_argument("--output", default="data/nsclc_subtype.csv")
    download.add_argument("--prepare-only", action="store_true")
    richer = sub.add_parser("download-rich", help="Download NSCLC Radiogenomics features and official labels")
    richer.add_argument("--raw", default="data/raw/nsclc_radiogenomics")
    richer.add_argument("--output", default="data/radiogenomics_subtype.csv")
    richer.add_argument("--prepare-only", action="store_true")
    crops = sub.add_parser("download-crops", help="Download verified TCGA crops and prepare CNN/texture features")
    crops.add_argument("--output", default="data/tcga_crops_features.csv")
    crops.add_argument("--prepare-only", action="store_true")
    experiment = sub.add_parser("experiment", help="Freeze stronger models using nested development CV")
    experiment.add_argument("--cohort", choices=["radiogenomics", "crops"], required=True)
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--data", help="Custom standardized radiomics CSV for the radiogenomics model comparison")
    experiment.add_argument("--evaluate", action="store_true", help="Evaluate the locked holdout once after freezing models")
    accuracy = sub.add_parser("accuracy", help="Accuracy-focused nested search on the preserved Radiogenomics split")
    accuracy.add_argument("--output", default="artifacts/radiogenomics_accuracy")
    accuracy.add_argument("--evaluate", action="store_true", help="Score the preserved, previously inspected holdout after freezing selection")
    expanded = sub.add_parser("accuracy-clinical", help="Add eligible clinical development patients while preserving the same holdout")
    expanded.add_argument("--output", default="artifacts/radiogenomics_expanded_clinical")
    expanded.add_argument("--evaluate", action="store_true")
    refinement = sub.add_parser("refine", help="Compare calibrated, bagged, sparse, PCA and stacked radiomics models")
    refinement.add_argument("--output", default="artifacts/radiogenomics_refinement")
    refinement.add_argument("--evaluate", action="store_true")
    training = sub.add_parser("train", help="Fit paper MLP and CatBoost; evaluate held-out patients")
    training.add_argument("--data", required=True)
    training.add_argument("--output", default="artifacts/latest")
    training.add_argument("--task", choices=["subtype", "malignancy"], default="subtype")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--cv", type=int, default=5)
    training.add_argument("--quick", action="store_true", help="Single parameter choice per model")
    training.add_argument("--synthetic", action="store_true")
    training.add_argument("--no-calibration", action="store_true")
    prediction = sub.add_parser("predict", help="Predict from extracted radiomic features")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--output", default="predictions.csv")
    extraction = sub.add_parser("extract", help="Extract 3D CT radiomics from an image/mask manifest")
    extraction.add_argument("--manifest", required=True)
    extraction.add_argument("--params", default="configs/radiomics.yaml")
    extraction.add_argument("--output", default="data/extracted_features.csv")
    serve = sub.add_parser("serve", help="Run the local research API on loopback")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--artifacts", default="artifacts")
    sub.add_parser("doctor", help="Inspect installed dependencies, models and source-data gates")
    profile = sub.add_parser("profile", help="Build input reference ranges from fitted training patients only")
    profile.add_argument("--model", required=True)
    profile.add_argument("--data", required=True)
    case_report = sub.add_parser("case-report", help="Export one case with explanation and quality checks")
    case_report.add_argument("--model", required=True)
    case_report.add_argument("--data", required=True)
    case_report.add_argument("--patient-id", required=True)
    case_report.add_argument("--output", default="case_report.html")
    external = sub.add_parser("validate-external", help="Evaluate a frozen classifier on a new independent cohort")
    external.add_argument("--model", required=True)
    external.add_argument("--data", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--source-evidence", required=True)
    inspect = sub.add_parser("inspect-ct", help="Inspect local CT geometry and export a tumour overlay")
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--mask", required=True)
    inspect.add_argument("--label", type=int, default=1)
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--stability", action="store_true")
    inspect.add_argument("--params", default="configs/radiomics.yaml")
    recurrence = sub.add_parser("recurrence", help="Audit, train, and predict first recurrence outcomes")
    recurrence_commands = recurrence.add_subparsers(dest="recurrence_command", required=True)
    recurrence_prepare = recurrence_commands.add_parser(
        "prepare", aliases=["audit"], help="Join the source cohort and audit recurrence outcome semantics")
    recurrence_prepare.add_argument("--features", default="data/radiogenomics_subtype.csv")
    recurrence_prepare.add_argument("--clinical", default="data/raw/nsclc_radiogenomics/clinical.csv")
    recurrence_prepare.add_argument("--output", default="data/recurrence_cohort.csv")
    recurrence_prepare.add_argument("--audit-dir", default="artifacts/recurrence_audit")
    recurrence_prepare.add_argument("--surgery-date-evidence",
                                    help="Documented source for interval direction and matching radiomic CT")
    recurrence_prepare.add_argument("--recurrence-followup-evidence",
                                    help="Documented source that recurrence-negative status is observed through censoring")
    recurrence_train = recurrence_commands.add_parser("train", help="Fit locked competing-risk recurrence models")
    recurrence_train.add_argument("--cohort", default="data/recurrence_cohort.csv")
    recurrence_train.add_argument("--output", default="artifacts/recurrence")
    recurrence_train.add_argument("--audit", default="artifacts/recurrence_audit/audit.json")
    recurrence_train.add_argument("--evaluate", action="store_true",
                                  help="Score the preserved legacy test patients once after freezing selection")
    recurrence_predict = recurrence_commands.add_parser("predict", help="Predict recurrence cumulative incidence")
    recurrence_predict.add_argument("--model", required=True)
    recurrence_predict.add_argument("--data", required=True)
    recurrence_predict.add_argument("--output", default="recurrence_predictions.csv")
    recorded_train = recurrence_commands.add_parser(
        "train-recorded", help="Fit an exploratory yes/no recorded-recurrence classifier; no time horizon")
    recorded_train.add_argument("--features", default="data/radiogenomics_clinical_subtype.csv")
    recorded_train.add_argument("--clinical", default="data/raw/nsclc_radiogenomics/clinical.csv")
    recorded_train.add_argument("--output", default="artifacts/recurrence_recorded")
    recorded_predict = recurrence_commands.add_parser(
        "predict-recorded", help="Predict source-recorded recurrence with a frozen classifier")
    recorded_predict.add_argument("--model", default="artifacts/recurrence_recorded/clinical_logistic.joblib")
    recorded_predict.add_argument("--data", required=True)
    recorded_predict.add_argument("--output", default="recorded_recurrence_predictions.csv")
    training_fit = recurrence_commands.add_parser(
        "training-fit", help="Overfit a small development subset and show training/test scores separately")
    training_fit.add_argument("--source", default="artifacts/recurrence_recorded")
    training_fit.add_argument("--output", default="artifacts/recurrence_training_fit")
    training_fit.add_argument("--patients", type=int, default=40)
    nodal = sub.add_parser("nodal", help="Audit, train and predict lymph-node involvement")
    nodal_commands = nodal.add_subparsers(dest="nodal_command", required=True)
    nodal_prepare = nodal_commands.add_parser("prepare", aliases=["audit"], help="Audit preoperative CT and pathological nodal labels")
    nodal_prepare.add_argument("--features", default="data/radiogenomics_subtype.csv")
    nodal_prepare.add_argument("--clinical", default="data/raw/nsclc_radiogenomics/clinical.csv")
    nodal_prepare.add_argument("--output", default="data/nodal_cohort.csv")
    nodal_prepare.add_argument("--audit-dir", default="artifacts/nodal_audit")
    nodal_prepare.add_argument("--surgery-date-evidence", help="Documented source establishing the matching radiomic CT and preoperative interval direction")
    nodal_train = nodal_commands.add_parser("train", help="Freeze nodal models using nested development CV")
    nodal_train.add_argument("--cohort", default="data/nodal_cohort.csv")
    nodal_train.add_argument("--audit", default="artifacts/nodal_audit/audit.json")
    nodal_train.add_argument("--output", default="artifacts/nodal")
    nodal_train.add_argument("--evaluate", action="store_true", help="Score previously inspected test patients after freezing selection")
    nodal_evaluate = nodal_commands.add_parser("evaluate", help="Evaluate a frozen nodal experiment once")
    nodal_evaluate.add_argument("--cohort", default="data/nodal_cohort.csv")
    nodal_evaluate.add_argument("--output", default="artifacts/nodal")
    nodal_predict = nodal_commands.add_parser("predict", help="Predict node-negative versus node-positive probability")
    nodal_predict.add_argument("--model", required=True)
    nodal_predict.add_argument("--data", required=True)
    nodal_predict.add_argument("--output", default="nodal_predictions.csv")
    args = parser.parse_args()
    if args.command == "download":
        from .datasets import download_lung1
        download_lung1(args.raw, args.output, download=not args.prepare_only)
    elif args.command == "download-rich":
        from .radiogenomics import download_radiogenomics
        download_radiogenomics(args.raw, args.output, args.prepare_only)
    elif args.command == "download-crops":
        from .ct_transfer import download_crops, prepare
        if not args.prepare_only:
            download_crops()
        prepare(args.output)
    elif args.command == "experiment":
        if args.cohort == "crops":
            if args.data:
                parser.error("--data is available for the radiogenomics comparison; crops use the verified crop dataset.")
            from .crop_experiment import run_crops
            run_crops(args.output, args.evaluate)
        else:
            from .rich_experiment import run
            run(data=args.data or "data/radiogenomics_subtype.csv", output=args.output, evaluate=args.evaluate)
    elif args.command == "accuracy":
        from .accuracy_experiment import prepare_clinical, run_accuracy
        data = prepare_clinical()
        run_accuracy(data, args.output, args.evaluate)
    elif args.command == "accuracy-clinical":
        from .accuracy_experiment import expanded_clinical_specs, prepare_expanded_clinical, run_accuracy
        data = prepare_expanded_clinical()
        run_accuracy(data, args.output, args.evaluate, specs=expanded_clinical_specs())
    elif args.command == "refine":
        from .refinement import run_refinement
        run_refinement(args.output, args.evaluate)
    elif args.command == "train":
        from .train import train
        train(args.data, args.output, args.task, args.seed, args.cv, args.quick,
              "synthetic" if args.synthetic else "real", not args.no_calibration)
    elif args.command == "predict":
        from .predict import load_bundle, predict_frame
        from .recurrence_predict import read_feature_csv
        from .dashboard_runtime import validate_queue
        frame, bundle = read_feature_csv(args.data), load_bundle(args.model)
        validate_queue(bundle, frame)
        if Path(args.output).resolve() in {Path(args.model).resolve(), Path(args.data).resolve()}:
            parser.error("Prediction output must not overwrite its model or input.")
        result = predict_frame(bundle, frame)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} predictions to {args.output}")
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        if not 1 <= args.port <= 65535:
            parser.error("Port must be between 1 and 65535.")
        uvicorn.run(create_app(args.artifacts), host="127.0.0.1", port=args.port)
    elif args.command == "doctor":
        from importlib.util import find_spec
        from .service import discover_models, extension_readiness
        models = discover_models()
        print(json.dumps({"dependencies": {name: find_spec(name) is not None for name in
            ["numpy", "pandas", "sklearn", "streamlit", "fastapi", "uvicorn", "SimpleITK", "radiomics"]},
            "registered_models": len(models), "extensions": extension_readiness(),
            "context": "Software readiness is separate from independent validation and verified outcomes."}, indent=2))
    elif args.command in {"profile", "case-report", "validate-external", "inspect-ct"}:
        try:
            if args.command == "profile":
                from .quality import create_profile
                result = create_profile(args.model, args.data)
                print(f"Saved training-only profile for {result['training_patients']} patients.")
            elif args.command == "validate-external":
                from .external_validation import validate_external
                print(json.dumps(validate_external(args.model, args.data, args.output, args.source_evidence), indent=2))
            elif args.command == "case-report":
                from .predict import load_bundle
                from .recurrence_predict import read_feature_csv
                from .service import run_prediction
                from .quality import assess_inputs, load_profile, sha256
                from .explanations import explain_case
                from .case_report import build_case_report
                frame = read_feature_csv(args.data)
                frame = frame.loc[frame.patient_id.eq(args.patient_id)]
                if len(frame) != 1:
                    raise ValueError("The requested patient ID must identify exactly one case.")
                bundle = load_bundle(args.model)
                profile = load_profile(args.model, bundle)
                result = run_prediction(bundle, frame)
                html = build_case_report(bundle, frame, result, sha256(args.model),
                    assess_inputs(bundle, frame, profile), explain_case(bundle, frame, profile))
                output = Path(args.output)
                if output.exists() or output.resolve() in {Path(args.model).resolve(), Path(args.data).resolve()}:
                    raise ValueError("Use a new report path; existing files cannot be overwritten.")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(html)
                print(f"Saved {output}")
            else:
                from .imaging import read_volume
                from .ct_workspace import inspect_volume, slice_figure, segmentation_stability
                image, mask = read_volume(args.image), read_volume(args.mask)
                output = Path(args.output)
                if output.exists():
                    raise ValueError("Use a new inspection directory.")
                quality = inspect_volume(image, mask, args.label)
                output.mkdir(parents=True)
                (output / "quality.json").write_text(json.dumps(quality, indent=2))
                slice_figure(image, mask, label=args.label).savefig(output / "overlay.png", dpi=150)
                if args.stability:
                    summary, features = segmentation_stability(image, mask, args.params, args.label)
                    summary.to_csv(output / "stability.csv", index=False)
                    features.to_csv(output / "stability_features.csv", index=False)
                print(f"Saved CT inspection to {output}")
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
    elif args.command == "nodal":
        try:
            if args.nodal_command in {"prepare", "audit"}:
                from .nodal_data import prepare_nodal
                result = prepare_nodal(args.features, args.clinical, args.output, args.audit_dir,
                                       args.surgery_date_evidence)
            elif args.nodal_command == "train":
                from .nodal_train import run_nodal
                result = run_nodal(args.cohort, args.output, audit_path=args.audit, evaluate=args.evaluate)
            elif args.nodal_command == "evaluate":
                from .nodal_train import evaluate_nodal
                result = evaluate_nodal(args.cohort, args.output)
            else:
                from .nodal_predict import predict_csv
                result = predict_csv(args.model, args.data, args.output)
                print(f"Saved {len(result)} nodal predictions to {args.output}")
                return
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
    elif args.command == "recurrence":
        if args.recurrence_command == "training-fit":
            from .recurrence_classifier import run_training_fit_experiment
            try:
                result = run_training_fit_experiment(args.source, args.output, args.patients)
            except (ValueError, OSError) as error:
                parser.error(str(error))
            print(json.dumps({key: result[key] for key in (
                "model", "training_patients", "test_patients", "training_metrics", "test_metrics", "interpretation")}, indent=2))
        elif args.recurrence_command == "train-recorded":
            from .recurrence_classifier import run_recorded_recurrence
            try:
                result = run_recorded_recurrence(args.features, args.clinical, args.output)
            except (ValueError, OSError) as error:
                parser.error(str(error))
            print(json.dumps({key: result[key] for key in (
                "endpoint", "development_patients", "test_patients", "selected_model_by_development",
                "development_metrics", "test_metrics", "prevalence_reference_test", "prediction_context")}, indent=2))
        elif args.recurrence_command == "predict-recorded":
            from .recurrence_classifier import predict_recorded_csv
            try:
                result = predict_recorded_csv(args.model, args.data, args.output)
            except (ValueError, OSError) as error:
                parser.error(str(error))
            print(f"Saved {len(result)} recorded-recurrence predictions to {args.output}")
        elif args.recurrence_command in {"prepare", "audit"}:
            from .recurrence_data import prepare_recurrence
            result = prepare_recurrence(
                args.features, args.clinical, args.output, args.audit_dir,
                args.surgery_date_evidence, args.recurrence_followup_evidence)
            print(json.dumps(result, indent=2))
        elif args.recurrence_command == "train":
            from .recurrence_train import run_recurrence
            try:
                result = run_recurrence(args.cohort, args.output, args.evaluate, args.audit)
            except ValueError as error:
                parser.error(str(error))
            print(json.dumps(result, indent=2, default=str))
        else:
            from .recurrence_predict import predict_csv
            result = predict_csv(args.model, args.data, args.output)
            print(f"Saved {len(result)} recurrence predictions to {args.output}")
    elif args.command == "extract":
        from .imaging import extract_manifest
        extract_manifest(args.manifest, args.params, args.output)


if __name__ == "__main__":
    main()
