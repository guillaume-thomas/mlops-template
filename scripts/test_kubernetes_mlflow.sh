docker login -u="${{vars.QUAY_ROBOT_USERNAME}}" -p="${{secrets.QUAY_ROBOT_TOKEN}}" quay.io
kubectl config set-cluster openshift-cluster --server=${{vars.OPENSHIFT_SERVER}}
kubectl config set-credentials openshift-credentials --token=${{secrets.OPENSHIFT_TOKEN}}
kubectl config set-context openshift-context --cluster=openshift-cluster --user=openshift-credentials --namespace=${{vars.OPENSHIFT_USERNAME}}-dev
kubectl config use openshift-context

export KUBE_MLFLOW_TRACKING_URI=http://mlflow-gthomas59800-dev.apps.rm3.7wse.p1.openshiftapps.com
export MLFLOW_TRACKING_URI=http://mlflow-gthomas59800-dev.apps.rm3.7wse.p1.openshiftapps.com
export MLFLOW_S3_ENDPOINT_URL=http://minio-api-gthomas59800-dev.apps.rm3.7wse.p1.openshiftapps.com
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123

docker build -f ./k8s/experiment/Dockerfile -t quay.io/gthomas59800/summittest:latest --build-arg MLFLOW_S3_ENDPOINT_URL=$MINIO_API_ROUTE_URL --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID--build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY ..

uv run mlflow run ../src/summit/training -P path=all_titanic.csv --experiment-name summittest --backend kubernetes --backend-config ../k8s/experiment/kubernetes_config.json
