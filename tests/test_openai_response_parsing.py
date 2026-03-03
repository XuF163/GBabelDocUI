import pytest


from pdf2zh_next.translator.translator_impl.openai import _extract_message_content_from_response


def test_extract_from_plain_string():
    assert _extract_message_content_from_response("  hello  ") == "hello"


def test_extract_from_json_string_content():
    assert _extract_message_content_from_response('{"content":" hi "}') == "hi"


def test_extract_from_openai_style_dict():
    resp = {"choices": [{"message": {"content": " ok "}}]}
    assert _extract_message_content_from_response(resp) == "ok"


def test_extract_from_siliconflow_style_dict():
    resp = {"content": " translated "}
    assert _extract_message_content_from_response(resp) == "translated"


def test_extract_raises_on_error_payload():
    with pytest.raises(RuntimeError):
        _extract_message_content_from_response({"detail": "unauthorized"})


def test_extract_from_object_with_choices():
    class Msg:
        content = " works "

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    assert _extract_message_content_from_response(Resp()) == "works"
