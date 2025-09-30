from mmpretrain import ImageClassificationInferencer
import torch


class MyClassificationInferencer(ImageClassificationInferencer):
    def __init__(self,
                 model,
                 pretrained=True,
                 device=None,
                 classes=None,
                 **kwargs) -> None:
        super().__init__(
            model=model, pretrained=pretrained, device=device, classes=classes, **kwargs)
        self.postprocess_kwargs = {'top'}

    def postprocess(self,
                    preds,
                    visualization,
                    return_datasamples=False,
                    top=1
                    ) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            pred_scores = data_sample.pred_score
            # Get the indices of the top k scores
            top_indices = torch.topk(pred_scores, k=top).indices
            # Extract information for each of the top k predictions
            results_top = []
            for i in range(top):
                pred_index = top_indices[i].item()
                pred_score = float(pred_scores[pred_index].item())

                result_top_sub = {
                    'pred_scores': pred_scores.detach().cpu().numpy(),
                    'pred_label': pred_index,
                    'pred_score': pred_score,
                }
                if self.classes is not None:
                    result_top_sub['pred_class'] = self.classes[pred_index]
                results_top.append(result_top_sub)
            results.append(results_top)

        return results


inferencer = MyClassificationInferencer(
    model='configs/resnet/jgwn2101.py',
    pretrained='/data/JGW/hsguan/mmpretrain/jgwdata_l_resnet101/epoch_400.pth',
    device='cuda:1')
result = inferencer(['web/壴30.png', 'web/安48.png'], top=20)
a = 0
