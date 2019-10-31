import pytest

from ..data_providers import DataProvider

import pandas as pd


@pytest.mark.parametrize(
    "batch_size, n_dat, hold_out, channelInds, return2D, rescale_to, crop_to, normalize_intensity",
    [
        [1, -1, 0.1, [0, 1, 2], False, None, None, False],
        [2, -1, 0.1, [0, 1, 2], False, None, None, False],
        [2, -1, 0.1, [0, 1, 2], True, None, None, False],
    ],
)
def test_dataprovider(
    data_dir,
    batch_size,
    n_dat,
    hold_out,
    channelInds,
    return2D,
    rescale_to,
    crop_to,
    normalize_intensity,
):
    csv_name = "test_data.csv"
    test_data_csv = "{}/{}".format(data_dir, csv_name)

    df = pd.read_csv(test_data_csv)

    dp = DataProvider.DataProvider(
        image_parent=data_dir,
        csv_name=csv_name,
        batch_size=batch_size,
        n_dat=n_dat,
        hold_out=hold_out,
        channelInds=channelInds,
        return2D=return2D,
        rescale_to=rescale_to,
        crop_to=crop_to,
        normalize_intensity=normalize_intensity,
    )

    n_dat = 0
    for group in ["train", "test", "validate"]:
        n_dat += dp.get_n_dat(group)

    assert n_dat == len(df)

    x, classes, ref = dp.get_sample()

    assert x.shape[0] == batch_size
    assert x.shape[1] == len(channelInds)

    if return2D:
        assert len(x.shape) == 4
    else:
        assert len(x.shape) == 5
