def cal_my_metric(ctx, **kwargs):
    return ctx["num_train_data"]


def call_my_metric(types):
    if "mymetric" in types:
        metric_builder = cal_my_metric
        return "mymetric", metric_builder