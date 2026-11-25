from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    deployments = [
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02h00
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
    ]
    serve(*deployments)
