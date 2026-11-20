"""Parsing of a stream request's frame epochs (plain epochs, runs, sequence)."""
from weather_radar.lib.frametimes import frame_times, MAX_FRAMES


class TestPlainLists:
    def test_batch_keeps_given_order(self):
        assert frame_times(None, None, 0, "30,10,20") == [30, 10, 20]

    def test_whitespace_empty_and_garbage_ignored(self):
        assert frame_times(None, None, 0, " 5, ,x, 7 ,1e3x,") == [5, 7]

    def test_none_and_empty(self):
        assert frame_times(None, None, 0, None) == []
        assert frame_times(None, None, 0, "") == []

    def test_duplicates_keep_first_occurrence(self):
        assert frame_times(None, None, 0, "5,9,5,9,7") == [5, 9, 7]


class TestRuns:
    def test_ascending_run_is_inclusive(self):
        assert frame_times(None, None, 0, "100:400:100") == [100, 200, 300, 400]

    def test_descending_run_counts_down(self):
        assert frame_times(None, None, 0, "400:100:100") == [400, 300, 200, 100]

    def test_step_sign_is_ignored_direction_comes_from_endpoints(self):
        assert frame_times(None, None, 0, "400:100:-100") == [400, 300, 200, 100]
        assert frame_times(None, None, 0, "100:400:-100") == [100, 200, 300, 400]

    def test_run_that_does_not_land_on_to_stops_before_it(self):
        assert frame_times(None, None, 0, "0:250:100") == [0, 100, 200]

    def test_single_point_run(self):
        assert frame_times(None, None, 0, "500:500:900") == [500]

    def test_mixed_runs_and_singles_preserve_order(self):
        got = frame_times(None, None, 0, "7,300:100:100,9,1000:1200:100")
        assert got == [7, 300, 200, 100, 9, 1000, 1100, 1200]

    def test_malformed_runs_are_skipped(self):
        assert frame_times(None, None, 0, "1:2,3:4:5:6,a:b:c,10:20:0,8") == [8]

    def test_overlap_with_singles_is_deduped(self):
        assert frame_times(None, None, 0, "200,100:300:100") == [200, 100, 300]

    def test_real_request_from_the_wild(self):
        # the 242-epoch backwards batch that motivated runs
        run = frame_times(None, None, 0, "1795547700:1795330800:900")
        assert len(run) == 242
        assert run[0] == 1795547700 and run[-1] == 1795330800
        assert all(a - b == 900 for a, b in zip(run, run[1:]))


class TestSequenceAndCap:
    def test_sequence_follows_the_batch(self):
        assert frame_times(1000, 1300, 100, "50,40") == [50, 40, 1000, 1100, 1200, 1300]

    def test_frames_in_both_are_sent_once(self):
        assert frame_times(1000, 1200, 100, "1100:1000:100") == [1100, 1000, 1200]

    def test_default_dt_is_hourly(self):
        assert frame_times(0, 7200, 0, None) == [0, 3600, 7200]

    def test_end_is_inclusive_with_one_second_slack(self):
        assert frame_times(0, 198, 100, None) == [0, 100]
        assert frame_times(0, 199, 100, None) == [0, 100, 200]   # end + 1 s slack
        assert frame_times(0, 200, 100, None) == [0, 100, 200]

    def test_total_is_capped(self):
        out = frame_times(0, 10**9, 1, "1000000:2000000:1")
        assert len(out) == MAX_FRAMES and out[0] == 1_000_000

    def test_huge_run_does_not_hang(self):
        assert len(frame_times(None, None, 0, "0:99999999999:1")) == MAX_FRAMES
