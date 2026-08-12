import pytest

from gateway import mt5_client
from gateway.mt5_client import Mt5Error, _retcode_reason
from tests.conftest import FakePendingOrder, FakePosition


def test_known_retcode_is_decoded():
    assert "autotrading disabled in the terminal" in _retcode_reason(10027)


def test_unknown_retcode_is_silent():
    assert _retcode_reason(99999) == ""


def test_none_retcode_is_silent():
    assert _retcode_reason(None) == ""


def test_filling_type_prefers_fok_when_supported(fake_mt5):
    fake_mt5.filling_mode = fake_mt5.SYMBOL_FILLING_FOK | fake_mt5.SYMBOL_FILLING_IOC
    assert mt5_client.client._filling_type("XAUUSD") == fake_mt5.ORDER_FILLING_FOK


def test_filling_type_falls_back_to_ioc(fake_mt5):
    fake_mt5.filling_mode = fake_mt5.SYMBOL_FILLING_IOC
    assert mt5_client.client._filling_type("XAUUSD") == fake_mt5.ORDER_FILLING_IOC


def test_filling_type_falls_back_to_return_when_neither_supported(fake_mt5):
    # e.g. a synthetic index whose broker only does exchange-style execution —
    # this is exactly what caused retcode=10030 with a hardcoded IOC.
    fake_mt5.filling_mode = 0
    assert mt5_client.client._filling_type("SYNTH_INDEX") == fake_mt5.ORDER_FILLING_RETURN


def test_order_send_passes_magic_through_to_the_request_and_echoes_it(fake_mt5, monkeypatch):
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    sent_requests = []
    original_order_send = fake_mt5.order_send

    def capture(request):
        sent_requests.append(request)
        return original_order_send(request)

    monkeypatch.setattr(fake_mt5, "order_send", capture)

    result = mt5_client.client.order_send("XAUUSD", "buy", 0.1, None, None, "", magic=777)

    assert sent_requests[-1]["magic"] == 777
    assert result["magic"] == 777


def test_positions_reads_native_mt5_magic_field(fake_mt5, monkeypatch):
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    fake_mt5._positions[555] = FakePosition(
        ticket=555,
        symbol="XAUUSD",
        type_=fake_mt5.ORDER_TYPE_BUY,
        volume=0.1,
        price_open=2400.0,
        sl=2390.0,
        tp=2410.0,
        time=1_752_100_812,
        profit=0.0,
        magic=321,
    )

    (position,) = mt5_client.client.positions()

    assert position["magic"] == 321


def test_normalize_volume_bumps_up_to_symbol_minimum(fake_mt5):
    # Synthetic indices reject anything below their (much coarser) minimum
    # with retcode=10014 (invalid volume) rather than adjusting it —
    # a forex-sized default of 0.01 should be bumped up to what's tradeable.
    fake_mt5.volume_min = 0.2
    fake_mt5.volume_max = 50.0
    fake_mt5.volume_step = 0.2
    assert mt5_client.client._normalize_volume("SYNTH_INDEX", 0.01) == 0.2


def test_normalize_volume_snaps_to_step(fake_mt5):
    fake_mt5.volume_min = 0.2
    fake_mt5.volume_max = 50.0
    fake_mt5.volume_step = 0.2
    assert mt5_client.client._normalize_volume("SYNTH_INDEX", 0.5) == 0.4


def test_normalize_volume_caps_at_symbol_maximum(fake_mt5):
    fake_mt5.volume_max = 10.0
    assert mt5_client.client._normalize_volume("XAUUSD", 25.0) == 10.0


def test_normalize_volume_leaves_valid_volume_untouched(fake_mt5):
    assert mt5_client.client._normalize_volume("XAUUSD", 0.5) == 0.5


def test_position_modify_no_changes_is_treated_as_success(fake_mt5, monkeypatch):
    # retcode=10025 ("no changes") means the requested sl/tp already match the
    # position — e.g. dragging a TP line back to where it started. The trade
    # server rejects it as a no-op, but the caller's desired state already
    # holds, so this should not surface as an error.
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    fake_mt5._positions[555] = FakePosition(
        ticket=555,
        symbol="XAUUSD",
        type_=fake_mt5.ORDER_TYPE_BUY,
        volume=0.1,
        price_open=2400.0,
        sl=2390.0,
        tp=2410.0,
        time=1_752_100_812,
        profit=0.0,
    )
    fake_mt5.reject_no_changes = True
    mt5_client.client.position_modify(555, sl=2390.0, tp=2410.0)


def test_position_modify_real_rejection_still_raises(fake_mt5, monkeypatch):
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    fake_mt5._positions[555] = FakePosition(
        ticket=555,
        symbol="XAUUSD",
        type_=fake_mt5.ORDER_TYPE_BUY,
        volume=0.1,
        price_open=2400.0,
        sl=2390.0,
        tp=2410.0,
        time=1_752_100_812,
        profit=0.0,
    )
    fake_mt5.reject_order = True
    with pytest.raises(Mt5Error, match="position_modify"):
        mt5_client.client.position_modify(555, sl=2395.0, tp=2410.0)


def test_modify_pending_order_no_changes_is_treated_as_success(fake_mt5, monkeypatch):
    # Same "no changes" retcode applies to pending-order modifies — e.g. a
    # chart drag released at (or snapped back to) the order's current price.
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    fake_mt5.add_pending_order(
        FakePendingOrder(
            ticket=8699799265,
            symbol="XAUUSD",
            type_=fake_mt5.ORDER_TYPE_BUY_LIMIT,
            volume_current=0.1,
            price_open=2400.0,
            sl=2390.0,
            tp=2410.0,
            time_setup=1_752_100_812,
        )
    )
    fake_mt5.reject_no_changes = True
    mt5_client.client.modify_pending_order(8699799265, price=2400.0, sl=2390.0, tp=2410.0)


def test_modify_pending_order_real_rejection_still_raises(fake_mt5, monkeypatch):
    monkeypatch.setattr(mt5_client.client, "_connected", True)
    fake_mt5.add_pending_order(
        FakePendingOrder(
            ticket=8699799265,
            symbol="XAUUSD",
            type_=fake_mt5.ORDER_TYPE_BUY_LIMIT,
            volume_current=0.1,
            price_open=2400.0,
            sl=2390.0,
            tp=2410.0,
            time_setup=1_752_100_812,
        )
    )
    fake_mt5.reject_order = True
    with pytest.raises(Mt5Error, match="modify_pending_order"):
        mt5_client.client.modify_pending_order(8699799265, price=2405.0, sl=2390.0, tp=2410.0)
