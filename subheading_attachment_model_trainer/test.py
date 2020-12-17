import argparse
from . import config as cfg
import grpc
from . import helper
import json
import os
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


def _grpc_predict(payload):
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "sh-pred"
    request.model_spec.signature_name = "serving_default"

    input_data = {}
    for instance in payload:
        for input_name, input_values in instance.items():
            if input_name not in input_data:
                input_data[input_name] = []
            input_data[input_name].append(input_values)

    for input_name, input_values in input_data.items():
        request.inputs[input_name].CopyFrom(tf.make_tensor_proto(input_values))

    result_future = stub.Predict.future(request)
    predictions = result_future.result().outputs["predictions"]
    predictions = tf.make_ndarray(predictions).astype("U").tolist()
    return predictions


def _print_metrics(y_true, y_pred):
    pred_counts = len(y_pred)
    act_counts = len(y_true)

    match_counts = 0
    for example_pair in y_pred:
        if example_pair in y_true:
            match_counts += 1

    p = match_counts / pred_counts
    r = match_counts / act_counts
    f = 2*p*r / (p + r + 1e-9)
    print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1 Score: {f:.3f}")


def run(workdir):
    batch_size = cfg.TEST_BATCH_SIZE
    critical_subheadings = cfg.CRITICAL_SUBHEADINGS
    encoding = cfg.ENCODING

    test_set_path = os.path.join(workdir, cfg.TEST_SET_FILENAME)
    test_set = helper.load_dataset(test_set_path, encoding)

    test_set = [json.loads(line) for line in test_set]
    y_true = { (str(example["pmid"]), dui, qui) 
               for example in test_set 
                    for dui, quis in example["mesh_headings"] 
                        for qui in quis if qui in critical_subheadings }
    test_set = [{ "pmid": str(example["pmid"]), 
                  "title": example["title"], 
                  "abstract": example["abstract"],
                  "pub_year": example["pub_year"],
                  "year_indexed": example["year_completed"],
                  "journal_id": example["journal_nlmid"],} 
                  for example in test_set]

    y_pred = []
    num_examples = len(test_set)
    for idx in range(0, len(test_set), batch_size):
        print(f"{idx:07d}/{num_examples:07d}", end="\r")
        batch_examples = test_set[idx:idx + batch_size]
        batch_predictions = _grpc_predict(batch_examples)
        batch_predictions = {(pmid, dui, qui) for pmid, dui, qui in batch_predictions if pmid and dui and qui}
        y_pred.extend(batch_predictions)
    print(f"{num_examples:07d}/{num_examples:07d}")

    _print_metrics(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", dest="workdir", help="The working directory.")
    args = parser.parse_args()
    workdir = args.workdir or os.getcwd()
    run(workdir)