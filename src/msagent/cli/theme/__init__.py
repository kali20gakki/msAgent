from msagent.cli.theme import tokyo_day, tokyo_night  # noqa: F401
from msagent.cli.theme.console import ThemedConsole
from msagent.cli.theme.detect import detect_terminal_theme
from msagent.cli.theme.registry import get_theme
from msagent.core.settings import settings

# Map detected theme mode to theme name
_THEME_MAP = {
    "dark": "tokyo-night",
    "light": "tokyo-day",
}

# Use user setting if set, otherwise auto-detect
if settings.cli.theme is not None:
    _theme_name = settings.cli.theme
else:
    _detected_mode = detect_terminal_theme()
    _theme_name = _THEME_MAP.get(_detected_mode, "tokyo-night")

theme = get_theme(_theme_name)
console = ThemedConsole(theme)
