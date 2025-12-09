from myst_parser.sphinx_ import MystParser
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumPyDocstring
from sphinx.ext.napoleon import Config

class NapoleonMySTParser(MystParser):
    """
    Custom parser that first processes docstrings with Napoleon (Google/NumPy style),
    then parses the resulting content (which may contain MyST Markdown) using MystParser.
    """
    def parse(self, input_string, document):
        # We need to determine if we should use Google or NumPy style.
        # Usually, one is dominant or both are enabled.
        # We can create a default Napoleon config.
        # Ideally, we would access the Sphinx app config, but the Parser API doesn't pass it directly easily
        # without digging into document.settings.env.app (if available).

        # Try to get config from document settings if possible, otherwise use defaults.
        # Napoleon Config defaults usually work for standard Google style.
        config = Config(napoleon_use_param=True, napoleon_use_rtype=True)

        # Convert Google style to RST (which is also valid MyST mostly)
        # Note: input_string might be a StringList or string. MystParser expects string usually?
        # docutils Parser.parse takes inputstring (str).

        # We try to treat it as GoogleDocstring.
        # If it's not strictly Google style, Napoleon usually leaves it mostly alone or just reformats sections.
        # Markdown tables in "Description" or non-standard sections should be preserved as text blocks.

        try:
            docstring = GoogleDocstring(input_string, config)
            # str(docstring) returns the converted RST.
            converted_string = str(docstring)
        except Exception:
            # Fallback if something goes wrong (e.g. unexpected input type)
            converted_string = input_string

        # Pass the converted string to the standard MyST parser
        return super().parse(converted_string, document)
