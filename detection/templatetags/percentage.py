from django import template

register = template.Library()


@register.filter
def percentage(value):
    """Convert decimal to percentage."""
    try:
        return f"{float(value) * 100:.1f}"
    except (ValueError, TypeError):
        return "0.0"

