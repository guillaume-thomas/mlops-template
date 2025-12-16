echo Uninstall kto-mlflow
oc delete -f k8s/mlflow/minio.yml
oc delete -f k8s/mlflow/mysql.yml
oc delete -f k8s/mlflow/mlflow.yml
oc delete -f k8s/mlflow/dailyclean.yml
oc delete -f k8s/api/api.yaml
oc delete -f k8s/monitoring/jaeger.yaml
oc delete -f k8s/chatbot/chatbot.yaml
oc delete -f k8s/mcp_server/mcp-server.yaml
oc delete secret chatbot-secrets
