# -*- coding: utf-8 -*-

class ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=False)
        return self.instance(*args, **kwargs)
