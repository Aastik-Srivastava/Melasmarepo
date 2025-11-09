import sys

# Import classes so they are available as modules
from .classification_model import FCMClassifier

try:
	# StablePNN may not exist unless pnn.py is present
	from .pnn import StablePNN
except Exception:
	StablePNN = None

# When pickles were created inside a notebook or a __main__ context, they
# reference classes by module '__main__'. At unpickle time Python looks up
# the class by name on that module. To avoid having to re-dump pickles,
# register the known classes on the running __main__ module so pickle can
# resolve them.
_main = sys.modules.get('__main__')
if _main is not None:
	try:
		setattr(_main, 'FCMClassifier', FCMClassifier)
	except Exception:
		pass
	if StablePNN is not None:
		try:
			setattr(_main, 'StablePNN', StablePNN)
		except Exception:
			pass

__all__ = [
	'FCMClassifier',
	'StablePNN',
]
