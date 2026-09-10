from src.tools.metadata_filter import allowed_access_levels, is_role_authorized, build_metadata_filter


def test_general_role_sees_only_general():
    assert allowed_access_levels("general") == ["general"]


def test_admin_sees_everything():
    levels = allowed_access_levels("admin")
    assert set(levels) == {"general", "admin"}


def test_unknown_role_defaults_to_general():
    assert allowed_access_levels("not_a_real_role") == ["general"]


def test_is_role_authorized():
    assert is_role_authorized("general", "general") is True
    assert is_role_authorized("general", "admin") is False


def test_build_metadata_filter_shape():
    assert build_metadata_filter("general") == {"access_level": {"$in": ["general"]}}
