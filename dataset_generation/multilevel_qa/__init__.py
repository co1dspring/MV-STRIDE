# multilevel_qa/__init__.py
from .level1_qa import Level1Mixin
from .level2_qa import Level2Mixin
# from .level3_reason import Level3Mixin
# from .base_generator import BaseQAProvider

# 最终合体类
class Multilevel_QA_Generator(Level1Mixin, Level2Mixin):
    def __init__(self, parent):
        self.parent = parent
