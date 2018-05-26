bash ./run_train.sh & \
while((1))
do
    sleep 600;
    bash ./run_eval.sh;
done;

bash ./export_to_inference.sh
