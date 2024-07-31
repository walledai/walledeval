# walledeval/prompts/core.py

from string import Template
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
import yaml, json
import enum
from typing import Optional, TypeVar

from walledeval.types import *

__all__ = [
    "AbstractPromptTemplate",
    "BasePromptTemplate", "BaseConversationTemplate", 
    "PromptTemplate", "Param"
]


class AbstractPromptTemplate(ABC):
    @abstractmethod
    def format(self, **kwds) -> str:
        raise NotImplementedError
    
    def __call__(self, **kwds) -> str:
        return self.format(**kwds)


class BasePromptTemplate(Template, AbstractPromptTemplate):
    """Basic Prompt Template Definition.

    Attributes:
        template (str): A string with $-delimited substitutions.
    """

    def __init__(self, template: str):
        """Inits BasePromptTemplate.

        Args:
            template (str): A string with $-delimited substitutions.
        """
        Template.__init__(self, template)

    def format(self, **kwds) -> str:
        """Formats Prompt Template with Keywords specified.

        Returns:
            str: Formatted (or partially formatted) template.
        """
        return self.safe_substitute(**kwds)


class BaseConversationTemplate(Template, AbstractPromptTemplate):
    """Basic Conversational Template Definition.
    
    Attributes:
        messages (Messages): A list of messages / list of Message objects / message
    """
    def __init__(self, messages: Messages):
        if isinstance(messages, list) and isinstance(messages[0], Message):
            messages = [
                dict(msg)
                for msg in messages
            ]
        elif isinstance(messages, str):
            messages = [{
                "role": "user",
                "message": messages
            }]
        
        self.messages = messages
        
        Template.__init__(self, json.dumps(self.messages))
        
        self.messages = [
            {
                "role": Template(message["role"]),
                "content": Template(message["content"])
            }
            for message in self.messages
        ]
    
    def format(self, **kwds) -> list[Message]:
        return [
            Message(
                role = message["role"].safe_substitute(**kwds),
                content = message["content"].safe_substitute(**kwds)
            ) for message in self.messages
        ]
        
        
        # return [
        #     Message(**it)
        #     for it in json.loads(self.safe_substitute(**kwds))
        # ]


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


class PromptTemplate(AbstractPromptTemplate):
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
    def from_yaml(cls, filename: str):
        yaml_fp = Path(filename)
        
        yaml_text = yaml_fp.read_text(encoding="utf-8")
        config = yaml.safe_load(yaml_text)
        
        prompt_type = config["type"]
        if prompt_type == "conversation":
            template = config["template"]
            if isinstance(template, str):
                template = template[:-1] if template.endswith("\n") else template
             #template = str(template)
        elif prompt_type == "prompt":
            template = config["template"]
            template = template[:-1] if template.endswith("\n") else template
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
            
            
            default = param.get("default", "None")
            
            if isinstance(default, list) and param.get("evaluate", False):
                default = [eval(i) for i in list]
            elif isinstance(default, str) and param.get("evaluate", False):
                default = eval(default)
        
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


    @classmethod
    def from_preset(cls, name: str = "mcq/default"):
        yaml_fp = Path(__file__).resolve().parent / f"presets/{name}.yaml"
        
        if yaml_fp.exists():
            return cls.from_yaml(str(yaml_fp))

    def format(self, input: object = None, **kwargs):
        params = {}
        
        for param in self.params:
            if param in kwargs:
                params[param] = kwargs[param]
            elif hasattr(input, param):
                params[param] = getattr(input, param)
            elif self.params[param].default is not None:
                params[param] = self.params[param].default
            # if not, it is not substituted
        
        final_params = {}
        
        for param in params:
            if any(i not in params for i in self.params[param].format_values):
                continue # leave this one blank too

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
    
    def __call__(self, input: object = None, **kwargs):
        return self.format(input, **kwargs)