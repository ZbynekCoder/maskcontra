from isanlp_rst.parser import Parser


class RSTParser:
    """
    Thin wrapper around isanlp_rst Parser. Fail fast on parsing issues.
    """

    def __init__(self, hf_model_name: str, hf_version: str, cuda_device: int = -1):
        self.parser = Parser(hf_model_name=hf_model_name, hf_model_version=hf_version, cuda_device=cuda_device)

    def save_rs3(self, text: str, out_path: str) -> None:
        res = self.parser(text)
        rst_list = res.get("rst", None)
        if not rst_list:
            raise RuntimeError("RST parser returned empty result")
        rst_list[0].to_rs3(out_path)
