# onnx_export.py
import torch
import onnx
from pathlib import Path
from omegaconf import DictConfig
import hydra
from dog_breed_detector.train.train import LitDogModel

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    model = LitDogModel(cfg)
    checkpoint = torch.load(cfg.exports.exports.model_path, map_location="cpu", weights_only=True)
        
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    
    model.eval()
    
    # Экспорт
    out_path = Path(cfg.exports.exports.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_input = torch.randn(1, 3, cfg.dataset.preprocessing.image_size, cfg.dataset.preprocessing.image_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=cfg.exports.exports.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} if cfg.exports.exports.dynamic_batch else None
    )
    
    # Валидация
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print(f"Успешно сохранили ONNX в {out_path}")

if __name__ == "__main__":
    main()