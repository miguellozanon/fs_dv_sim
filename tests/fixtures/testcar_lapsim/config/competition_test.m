function comp = competition_test()
% FICTIONAL competition of the CI test car: invented tracks (tracks/make_tracks.py) and invented scoring
% constants. Only for the automated tests.

here = fileparts(mfilename('fullpath'));
tdir = fullfile(here, '..', 'tracks');

comp.name = 'Test car, fictional tracks and scoring';

e = base_event('acceleration');
e.track   = lts_track_basic('acceleration', 75, 0.25);
e.util_x  = 1.0;  e.util_y = 1.0;
e.v0      = 0;
e.score   = score_def(50, 2.5, 4.0, 6.0, 2);
comp.events.acceleration = e;

e = base_event('skidpad');
e.track   = lts_track_basic('skidpad', 15.25/2 + 1.5, 0.25);
e.util_x  = 1.0;  e.util_y = 0.8;
e.timing  = 'flying_lap';
e.score   = score_def(50, 2.5, 5.0, 6.5, 2);
comp.events.skidpad = e;

e = base_event('autocross');
e.track   = lts_track_csv(fullfile(tdir, 'TEST_autocross.csv'), ...
                struct('closed', false, 'ds', 0.5, 'name', 'Test autocross'));
e.util_x  = 0.9;  e.util_y = 0.8;
e.v0      = 0;
e.score   = score_def(100, 5, 25.0, 40.0, 2);
comp.events.autocross = e;

e = base_event('endurance');
e.track   = lts_track_csv(fullfile(tdir, 'TEST_endurance.csv'), ...
                struct('closed', true, 'ds', 0.5, 'name', 'Test endurance'));
e.util_x  = 0.9;  e.util_y = 0.8;
e.laps    = 10;
e.timing  = 'flying_lap';
e.score   = score_def(250, 25, 330.0, 500.0, 2);
comp.events.endurance = e;
end

function e = base_event(type)
e.type   = type;
e.track  = struct();
e.util_x = 1;
e.util_y = 1;
e.v0     = NaN;
e.laps   = 1;
e.timing = 'single';
e.score  = struct();
end

function s = score_def(Pmax, Pmin, Tmin, Tmax, n)
s = struct('Pmax', Pmax, 'Pmin', Pmin, 'Tmin', Tmin, 'Tmax', Tmax, 'n', n);
end
