from qtpy.QtWidgets import QLayout, QLayoutItem


def _check_in_list(_values, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *_values*;
    if not, raise an appropriate ValueError.

    Examples
    --------
    >>> _check_in_list(["foo", "bar"], foo='foo', other_arg='other_val')
    Traceback (most recent call last):
    ...
    ValueError: other_val is not a valid value for other_arg; supported values are 'foo', 'bar'
    """
    values = _values
    for k, v in kwargs.items():
        if v not in values:
            raise ValueError(
                f"{v} is not a valid value for {k}; supported values are "
                f"{', '.join(map(repr, values))}"
            )


def clearLayout(layout: QLayout) -> None:
    if layout.count() == 0:
        return

    item: QLayoutItem = layout.takeAt(0)
    while item is not None:
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            item.layout().deleteLater()

        item: QLayoutItem = layout.takeAt(0)
