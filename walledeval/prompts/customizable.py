# walledeval/prompts/customizable.py

from pydantic import BaseModel
from pathlib import Path
import yaml
import enum
from typing import Optional, TypeVar

from pydantic import BaseModel

from walledeval.types import Messages

from walledeval.prompts.core import BasePromptTemplate, BaseConversationTemplate


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


class TemplateType(str, enum.Enum):
    PROMPT = "prompt"
    CONVERSATION = "conversation"


class CustomizableTemplate:
    def __init__(self, template: Messages = "$prompt",
                 type: TemplateType = TemplateType.PROMPT,
                 required: dict[str, Param] = dict(
                     prompt=Param(
                         name="prompt", 
                         param_type=str,
                         format_values=["prompt"])
                 ),
                 optional: dict[str, Param] = {},
                 **kwargs):
        # super().__init__(template)
        self.type = type
        
        if type == "prompt":
            self.template = BasePromptTemplate(template)
        else:
            self.template = BaseConversationTemplate(template)
        
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
        
        prompt_type = config["type"]
        if prompt_type == "conversation":
            template = config["template"]
            if isinstance(template, str):
                template = template.rstrip("\n")
            #template = str(template)
        elif prompt_type == "prompt":
            template = config["template"].rstrip("\n")
        else:
            raise ValueError(f"No such type '{prompt_type}', select from ['prompt', 'conversation']")
        
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
            TemplateType(prompt_type),
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
        
        return self.template.format(**final_params)