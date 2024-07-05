# walledeval/prompts/customizable.py

from pydantic import BaseModel
from pathlib import Path
import yaml
from typing import Optional, TypeVar

from pydantic import BaseModel

from walledeval.prompts.core import BasePromptTemplate


T = TypeVar('T')


class Param(BaseModel):
    name: str
    param_type: type
    default: Optional[T] = None
    format_func: str = "format = lambda it: it"
    format_values: list[str] = []

    
def _exec_with_return(code: str):
    lcls = locals()
    exec(code, globals(), lcls)
    format = lcls["format"]
    return format


class CustomizableTemplate(BasePromptTemplate):
    def __init__(self, template: str = "$prompt", 
                 required: dict[str, Param] = dict(
                     prompt=Param(
                         name="prompt", 
                         param_type=str,
                         format_values=["prompt"])
                 ),
                 optional: dict[str, Param] = {},
                 **kwargs):
        super().__init__(template)
        
        self.required = required
        self.optional = optional
        self.params = {**required, **optional}
        
        for key in optional:
            if key in kwargs:
                optional[key].default = kwargs[key]
            
        
        
    @classmethod
    def from_preset(cls, name: str = "mcq/default"):
        yaml_fp = Path(__file__).resolve().parent / f"presets/{name}.yaml"
        yaml_text = yaml_fp.read_text(encoding="utf-8")
        config = yaml.safe_load(yaml_text)

        template = config["template"].rstrip("\n")
        
        params = config["params"]

        optional_params = {}
        required_params = {}

        for param in params:
            name = param["name"]
            param_type = eval(param["type"])

            if "format" in param:
                format_func = param["format"]
                format_values = param["format_values"]
            else:
                format_func = f"format = lambda {name}: {name}"
                format_values = [name]
        
            param_config = Param(
                name = name,
                param_type = param_type,
                default = param.get("default", None),
                format_func = format_func,
                format_values=format_values
            )
        
            if param.get("optional", False):
                optional_params[name] = param_config
            else:
                required_params[name] = param_config
        
        return cls(
            template,
            required_params,
            optional_params
        )

    def format(self, input: object, **kwargs):
        params = {}
        
        for param in self.params:
            if param in kwargs:
                params[param] = kwargs[param]
            elif hasattr(input, param):
                params[param] = getattr(input, param)
            else:
                params[param] = self.params[param].default
        
        final_params = {}
        
        for param in self.params:
            format_func =_exec_with_return(
                self.params[param].format_func
            )
            
            variables = [
                params[i] 
                for i in self.params[param].format_values
            ]
            final_params[param] = format_func(
                *variables
            )
        
        return super().format(**final_params)