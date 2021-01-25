def _check_in_list(_values, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *_values*;
    if not, raise an appropriate ValueError.

    Examples
    --------
    >>> _check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    values = _values
    for k, v in kwargs.items():
        if v not in values:
            raise ValueError(
                f"{v} is not a valid value for {k}; supported values are "
                f"{', '.join(map(repr, values))}"
            )
