# Option A matched general-RL and ecological safe comparison

This table keeps all species, hidden families, noise levels, and methods separate.
Amur tiger is labelled declining/recoverable, Crab-eating fox strong-growth/recoverable,
and Egyptian vulture a demographic sink. No yield-mode claim is made.

## Acceptance and outcome-access chronology

- Original gates recorded `return_fields_opened=false`.
- Existing PLUS and scoped MOOR outcomes were subsequently opened under authorization.
- The two new method-specific Option A acceptances passed before this builder opened
  the new episode-level outcomes.
- Outcome opening time: `2026-07-23T21:23:21.625640+10:00`.

## Amur tiger — declining recoverable

| environment | sigma_obs | method | operational_return_mean | collapse_rate | unsafe_fraction | mvp_breach_rate | persistence_mean | min_population_mean | final_population_mean | economic_cost_mean | action_entropy_mean | constant_policy_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ricker | 0.1 | refplan | 2.00922 | 0 | 0 | 0 | 1 | 176.736 | 318.757 | 19.1275 | 1.65584 | nonconstant_observed |
| ricker | 0.1 | ogsrl | 4.17325 | 0 | 0 | 0 | 1 | 104.157 | 112.256 | 8.45312 | 1.03456 | nonconstant_observed |
| ricker | 0.1 | bamcts | 1.10466 | 0 | 0 | 0 | 1 | 197.288 | 375.924 | 23.5731 | 1.42022 | nonconstant_observed |
| ricker | 0.1 | ensemble_value_disagreement_pessimism | -13.8314 | 1 | 0.288235 | 1 | 0 | 11.7072 | 11.7072 | 4.52312 | 0.929869 | nonconstant_observed |
| ricker | 0.1 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| ricker | 0.1 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| ricker | 0.2 | refplan | 2.1677 | 0 | 0 | 0 | 1 | 175.379 | 341.314 | 19.1794 | 1.38683 | nonconstant_observed |
| ricker | 0.2 | ogsrl | 4.31236 | 0 | 0 | 0 | 1 | 103.581 | 117.693 | 7.74875 | 1.28412 | nonconstant_observed |
| ricker | 0.2 | bamcts | 1.29206 | 0 | 0 | 0 | 1 | 199.816 | 403.556 | 22.8125 | 0.950824 | nonconstant_observed |
| ricker | 0.2 | ensemble_value_disagreement_pessimism | -15.6383 | 1 | 0.307843 | 1 | 0 | 9.71201 | 9.71201 | 4.7875 | 1.03646 | nonconstant_observed |
| ricker | 0.2 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| ricker | 0.2 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| allee | 0.1 | refplan | 3.17422 | 0 | 0 | 0 | 1 | 176.895 | 439.755 | 17.1425 | 1.8937 | nonconstant_observed |
| allee | 0.1 | ogsrl | 2.91196 | 0 | 0 | 0 | 1 | 99.4146 | 105.798 | 8.76437 | 0.792475 | nonconstant_observed |
| allee | 0.1 | bamcts | 1.22918 | 0 | 0 | 0 | 1 | 199.43 | 497.199 | 25.9469 | 1.55343 | nonconstant_observed |
| allee | 0.1 | ensemble_value_disagreement_pessimism | -13.0908 | 1 | 0.272549 | 1 | 0 | 9.90779 | 9.90779 | 4.49938 | 1.12443 | nonconstant_observed |
| allee | 0.1 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| allee | 0.1 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| allee | 0.2 | refplan | 3.45447 | 0 | 0 | 0 | 1 | 145.029 | 369.911 | 14.4481 | 1.69514 | nonconstant_observed |
| allee | 0.2 | ogsrl | 4.7163 | 0 | 0 | 0 | 1 | 92.047 | 103.48 | 6.32812 | 1.13894 | nonconstant_observed |
| allee | 0.2 | bamcts | 1.27809 | 0 | 0 | 0 | 1 | 200 | 428.941 | 24.0944 | 1.14487 | nonconstant_observed |
| allee | 0.2 | ensemble_value_disagreement_pessimism | -11.6373 | 1 | 0.252941 | 1 | 0 | 9.14645 | 9.14645 | 3.79938 | 1.19685 | nonconstant_observed |
| allee | 0.2 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| allee | 0.2 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| theta | 0.1 | refplan | 2.01427 | 0 | 0 | 0 | 1 | 167.968 | 289.882 | 18.3544 | 1.90567 | nonconstant_observed |
| theta | 0.1 | ogsrl | 4.16314 | 0 | 0 | 0 | 1 | 109.132 | 119.218 | 6.835 | 0.764432 | nonconstant_observed |
| theta | 0.1 | bamcts | -0.431351 | 0 | 0 | 0 | 1 | 199.124 | 449.901 | 29.3625 | 1.6032 | nonconstant_observed |
| theta | 0.1 | ensemble_value_disagreement_pessimism | -11.2771 | 1 | 0.242157 | 1 | 0 | 14.8707 | 14.8707 | 6.21125 | 1.03197 | nonconstant_observed |
| theta | 0.1 | plus_adapted_ricker_only_pbvi | 4.2642 | 0 | 0 | 0 | 1 | 200 | 411.325 | 15.625 | 0 | constant_within_every_episode |
| theta | 0.1 | moor_adapted_ricker_misspec_pbvi | 4.2642 | 0 | 0 | 0 | 1 | 200 | 411.325 | 15.625 | 0 | constant_within_every_episode |
| theta | 0.2 | refplan | 2.35905 | 0 | 0 | 0 | 1 | 162.335 | 307.802 | 17.5594 | 1.63967 | nonconstant_observed |
| theta | 0.2 | ogsrl | 3.2175 | 0 | 0 | 0 | 1 | 83.822 | 106.525 | 7.98187 | 1.02294 | nonconstant_observed |
| theta | 0.2 | bamcts | 1.27287 | 0 | 0 | 0 | 1 | 199.363 | 391.288 | 24.0031 | 1.35497 | nonconstant_observed |
| theta | 0.2 | ensemble_value_disagreement_pessimism | -11.7126 | 1 | 0.243137 | 1 | 0 | 14.8291 | 14.8291 | 5.61187 | 1.01554 | nonconstant_observed |
| theta | 0.2 | plus_adapted_ricker_only_pbvi | 4.2642 | 0 | 0 | 0 | 1 | 200 | 411.325 | 15.625 | 0 | constant_within_every_episode |
| theta | 0.2 | moor_adapted_ricker_misspec_pbvi | 4.2642 | 0 | 0 | 0 | 1 | 200 | 411.325 | 15.625 | 0 | constant_within_every_episode |
| regime | 0.1 | refplan | 1.26132 | 0 | 0 | 0 | 1 | 184.169 | 456.394 | 23.5712 | 1.9238 | nonconstant_observed |
| regime | 0.1 | ogsrl | 2.61183 | 0 | 0 | 0 | 1 | 101.138 | 108.48 | 9.945 | 0.900156 | nonconstant_observed |
| regime | 0.1 | bamcts | -0.782497 | 0 | 0 | 0 | 1 | 200 | 495.631 | 32.1906 | 1.43165 | nonconstant_observed |
| regime | 0.1 | ensemble_value_disagreement_pessimism | -12.0462 | 1 | 0.248039 | 1 | 0 | 3.3708 | 3.3708 | 2.88875 | 1.1076 | nonconstant_observed |
| regime | 0.1 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| regime | 0.1 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| regime | 0.2 | refplan | 1.61081 | 0 | 0 | 0 | 1 | 180.497 | 451.4 | 22.5206 | 1.61825 | nonconstant_observed |
| regime | 0.2 | ogsrl | 2.75064 | 0 | 0 | 0 | 1 | 79.2623 | 106.68 | 8.93688 | 1.11254 | nonconstant_observed |
| regime | 0.2 | bamcts | 0.360453 | 0 | 0 | 0 | 1 | 199.552 | 490.453 | 26.8944 | 1.19333 | nonconstant_observed |
| regime | 0.2 | ensemble_value_disagreement_pessimism | -12.0119 | 1 | 0.237255 | 1 | 0 | 7.18984 | 7.18984 | 4.55313 | 1.15423 | nonconstant_observed |
| regime | 0.2 | plus_adapted_ricker_only_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |
| regime | 0.2 | moor_adapted_ricker_misspec_pbvi | 4.22442 | 0 | 0 | 0 | 1 | 200 | 403.795 | 15.625 | 0 | constant_within_every_episode |

## Crab-eating fox — strong growth recoverable

| environment | sigma_obs | method | operational_return_mean | collapse_rate | unsafe_fraction | mvp_breach_rate | persistence_mean | min_population_mean | final_population_mean | economic_cost_mean | action_entropy_mean | constant_policy_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ricker | 0.1 | refplan | 9.93748 | 0 | 0 | 1 | 1 | 38.8851 | 41.463 | -2.47938 | 1.14497 | nonconstant_observed |
| ricker | 0.1 | ogsrl | 10.1306 | 0 | 0 | 1 | 1 | 21.3544 | 21.4186 | -4.9975 | 0.00490196 | nonconstant_observed |
| ricker | 0.1 | bamcts | 9.61315 | 0 | 0 | 1 | 1 | 40.4091 | 60.2389 | 0.168125 | 1.50297 | nonconstant_observed |
| ricker | 0.1 | ensemble_value_disagreement_pessimism | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| ricker | 0.1 | plus_adapted_ricker_only_pbvi | 10.3956 | 0 | 0 | 1 | 1 | 37.0148 | 39.2412 | -3.565 | 0.676247 | nonconstant_observed |
| ricker | 0.1 | moor_adapted_ricker_misspec_pbvi | 10.386 | 0 | 0 | 1 | 1 | 37.2915 | 39.0715 | -3.5625 | 0.678809 | nonconstant_observed |
| ricker | 0.2 | refplan | 10.0238 | 0 | 0 | 1 | 1 | 38.1147 | 39.5994 | -2.87563 | 1.08341 | nonconstant_observed |
| ricker | 0.2 | ogsrl | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| ricker | 0.2 | bamcts | 9.61718 | 0 | 0 | 1 | 1 | 40.2665 | 56.3119 | 0.0075 | 1.51274 | nonconstant_observed |
| ricker | 0.2 | ensemble_value_disagreement_pessimism | 10.1244 | 0 | 0 | 1 | 1 | 21.3394 | 21.3394 | -4.985 | 0.0132992 | nonconstant_observed |
| ricker | 0.2 | plus_adapted_ricker_only_pbvi | 10.3841 | 0 | 0 | 1 | 1 | 33.5611 | 36.1646 | -4.0425 | 0.645915 | nonconstant_observed |
| ricker | 0.2 | moor_adapted_ricker_misspec_pbvi | 10.3856 | 0 | 0 | 1 | 1 | 34.2927 | 37.1188 | -3.9675 | 0.660537 | nonconstant_observed |
| allee | 0.1 | refplan | 6.74352 | 0.05 | 0.00392157 | 1 | 1 | 33.3597 | 78.9697 | 8.06688 | 2.01008 | nonconstant_observed |
| allee | 0.1 | ogsrl | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| allee | 0.1 | bamcts | 10.6966 | 0 | 0 | 1 | 1 | 38.773 | 81.4108 | 0.12625 | 1.29341 | nonconstant_observed |
| allee | 0.1 | ensemble_value_disagreement_pessimism | 10.2692 | 0 | 0 | 1 | 1 | 27.5709 | 32.7015 | -4.8375 | 0.239282 | nonconstant_observed |
| allee | 0.1 | plus_adapted_ricker_only_pbvi | 11.3942 | 0 | 0 | 1 | 1 | 40.1261 | 49.2461 | -3.0025 | 0.456887 | nonconstant_observed |
| allee | 0.1 | moor_adapted_ricker_misspec_pbvi | 11.7107 | 0 | 0 | 1 | 1 | 41 | 46.5431 | -3.1875 | 0.249757 | nonconstant_observed |
| allee | 0.2 | refplan | 7.94814 | 0 | 0 | 1 | 1 | 36.569 | 80.7593 | 8.71375 | 1.97367 | nonconstant_observed |
| allee | 0.2 | ogsrl | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| allee | 0.2 | bamcts | 10.2014 | 0 | 0 | 1 | 1 | 39.708 | 83.1563 | 1.79187 | 1.49684 | nonconstant_observed |
| allee | 0.2 | ensemble_value_disagreement_pessimism | 10.3974 | 0 | 0 | 1 | 1 | 29.6496 | 36.6075 | -4.73062 | 0.330102 | nonconstant_observed |
| allee | 0.2 | plus_adapted_ricker_only_pbvi | 10.9667 | 0 | 0 | 1 | 1 | 37.3813 | 55.0964 | -3.2875 | 0.448599 | nonconstant_observed |
| allee | 0.2 | moor_adapted_ricker_misspec_pbvi | 11.6109 | 0 | 0 | 1 | 1 | 40.3153 | 46.9413 | -3.15563 | 0.258507 | nonconstant_observed |
| theta | 0.1 | refplan | 9.54941 | 0 | 0 | 1 | 1 | 40.2707 | 79.496 | 3.1025 | 1.76473 | nonconstant_observed |
| theta | 0.1 | ogsrl | 10.1421 | 0 | 0 | 1 | 1 | 21.7949 | 22.6209 | -4.9875 | 0.0245098 | nonconstant_observed |
| theta | 0.1 | bamcts | 10.614 | 0 | 0 | 1 | 1 | 40.8344 | 76.7511 | -0.688125 | 1.33792 | nonconstant_observed |
| theta | 0.1 | ensemble_value_disagreement_pessimism | 10.1374 | 0 | 0 | 1 | 1 | 21.4319 | 21.8908 | -4.995 | 0.00980391 | nonconstant_observed |
| theta | 0.1 | plus_adapted_ricker_only_pbvi | 10.2202 | 0 | 0 | 1 | 1 | 23.6692 | 25.5549 | -4.9425 | 0.0827171 | nonconstant_observed |
| theta | 0.1 | moor_adapted_ricker_misspec_pbvi | 10.1465 | 0 | 0 | 1 | 1 | 21.6839 | 21.6839 | -4.95 | 0.0980391 | nonconstant_observed |
| theta | 0.2 | refplan | 9.61 | 0 | 0 | 1 | 1 | 39.3947 | 69.955 | 1.4175 | 1.64796 | nonconstant_observed |
| theta | 0.2 | ogsrl | 10.1633 | 0 | 0 | 1 | 1 | 22.8111 | 24.0142 | -4.9725 | 0.0539215 | nonconstant_observed |
| theta | 0.2 | bamcts | 10.4483 | 0 | 0 | 1 | 1 | 40.5311 | 72.0429 | -0.71125 | 1.38389 | nonconstant_observed |
| theta | 0.2 | ensemble_value_disagreement_pessimism | 10.1374 | 0 | 0 | 1 | 1 | 21.4039 | 21.4039 | -5 | 0 | constant_within_every_episode |
| theta | 0.2 | plus_adapted_ricker_only_pbvi | 10.3114 | 0 | 0 | 1 | 1 | 26.2527 | 36.5571 | -4.78 | 0.292573 | nonconstant_observed |
| theta | 0.2 | moor_adapted_ricker_misspec_pbvi | 10.1374 | 0 | 0 | 1 | 1 | 21.4039 | 21.4039 | -5 | 0 | constant_within_every_episode |
| regime | 0.1 | refplan | 6.84225 | 0.15 | 0.0117647 | 1 | 1 | 23.1182 | 77.3209 | 8.965 | 1.95062 | nonconstant_observed |
| regime | 0.1 | ogsrl | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| regime | 0.1 | bamcts | 10.0752 | 0 | 0 | 1 | 1 | 37.0579 | 79.7821 | 1.715 | 1.46878 | nonconstant_observed |
| regime | 0.1 | ensemble_value_disagreement_pessimism | 10.2874 | 0 | 0 | 1 | 1 | 27.5197 | 37.3024 | -4.675 | 0.380646 | nonconstant_observed |
| regime | 0.1 | plus_adapted_ricker_only_pbvi | 10.4543 | 0 | 0 | 1 | 1 | 36.9452 | 54.2473 | -2.3675 | 0.665381 | nonconstant_observed |
| regime | 0.1 | moor_adapted_ricker_misspec_pbvi | 10.1531 | 0 | 0 | 1 | 1 | 24.1649 | 24.1649 | -4.45938 | 0.0980391 | nonconstant_observed |
| regime | 0.2 | refplan | 4.31303 | 0.2 | 0.0196078 | 1 | 0.95 | 22.9942 | 69.0402 | 9.60562 | 1.93451 | nonconstant_observed |
| regime | 0.2 | ogsrl | 10.1304 | 0 | 0 | 1 | 1 | 21.2971 | 21.2971 | -5 | 0 | constant_within_every_episode |
| regime | 0.2 | bamcts | 9.37157 | 0 | 0 | 1 | 1 | 33.0945 | 77.3189 | 4.62563 | 1.65132 | nonconstant_observed |
| regime | 0.2 | ensemble_value_disagreement_pessimism | 10.3055 | 0 | 0 | 1 | 1 | 27.9995 | 38.4014 | -4.6225 | 0.417408 | nonconstant_observed |
| regime | 0.2 | plus_adapted_ricker_only_pbvi | 10.5907 | 0 | 0 | 1 | 1 | 37.912 | 53.8342 | -2.3475 | 0.73807 | nonconstant_observed |
| regime | 0.2 | moor_adapted_ricker_misspec_pbvi | 10.1425 | 0 | 0 | 1 | 1 | 24.2871 | 24.2871 | -4.41938 | 0.111277 | nonconstant_observed |

## Egyptian vulture — sink

| environment | sigma_obs | method | operational_return_mean | collapse_rate | unsafe_fraction | mvp_breach_rate | persistence_mean | min_population_mean | final_population_mean | economic_cost_mean | action_entropy_mean | constant_policy_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ricker | 0.1 | refplan | -186.816 | 0 | 1 | 1 | 0 | 9.64218 | 19.1772 | 10.7763 | 1.77646 | nonconstant_observed |
| ricker | 0.1 | ogsrl | -188.368 | 0 | 1 | 1 | 0 | 40.96 | 42.6769 | 15.9688 | 0.140471 | nonconstant_observed |
| ricker | 0.1 | bamcts | -189.608 | 0 | 1 | 1 | 0 | 8.19992 | 13.2851 | 16.2088 | 2.18937 | nonconstant_observed |
| ricker | 0.1 | ensemble_value_disagreement_pessimism | -187.839 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.15625 | 0.868833 | nonconstant_observed |
| ricker | 0.1 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| ricker | 0.1 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| ricker | 0.2 | refplan | -187.239 | 0 | 1 | 1 | 0 | 11.929 | 20.2799 | 11.93 | 1.75218 | nonconstant_observed |
| ricker | 0.2 | ogsrl | -188.685 | 0 | 1 | 1 | 0 | 40.9115 | 42.4505 | 16.8313 | 0.277578 | nonconstant_observed |
| ricker | 0.2 | bamcts | -190.351 | 0 | 1 | 1 | 0 | 3.62191 | 6.3965 | 17.2125 | 2.29673 | nonconstant_observed |
| ricker | 0.2 | ensemble_value_disagreement_pessimism | -187.818 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.1375 | 0.868006 | nonconstant_observed |
| ricker | 0.2 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| ricker | 0.2 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| allee | 0.1 | refplan | -186.816 | 0 | 1 | 1 | 0 | 9.64218 | 19.1772 | 10.7763 | 1.77646 | nonconstant_observed |
| allee | 0.1 | ogsrl | -188.368 | 0 | 1 | 1 | 0 | 40.96 | 42.6769 | 15.9688 | 0.140471 | nonconstant_observed |
| allee | 0.1 | bamcts | -189.608 | 0 | 1 | 1 | 0 | 8.19992 | 13.2851 | 16.2088 | 2.18937 | nonconstant_observed |
| allee | 0.1 | ensemble_value_disagreement_pessimism | -187.839 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.15625 | 0.868833 | nonconstant_observed |
| allee | 0.1 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| allee | 0.1 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| allee | 0.2 | refplan | -187.239 | 0 | 1 | 1 | 0 | 11.929 | 20.2799 | 11.93 | 1.75218 | nonconstant_observed |
| allee | 0.2 | ogsrl | -188.685 | 0 | 1 | 1 | 0 | 40.9115 | 42.4505 | 16.8313 | 0.277578 | nonconstant_observed |
| allee | 0.2 | bamcts | -190.351 | 0 | 1 | 1 | 0 | 3.62191 | 6.3965 | 17.2125 | 2.29673 | nonconstant_observed |
| allee | 0.2 | ensemble_value_disagreement_pessimism | -187.818 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.1375 | 0.868006 | nonconstant_observed |
| allee | 0.2 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| allee | 0.2 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| theta | 0.1 | refplan | -186.768 | 0 | 1 | 1 | 0 | 10.2733 | 19.5737 | 10.895 | 1.8039 | nonconstant_observed |
| theta | 0.1 | ogsrl | -188.258 | 0 | 1 | 1 | 0 | 41 | 44.7513 | 15.8562 | 0.09353 | nonconstant_observed |
| theta | 0.1 | bamcts | -189.796 | 0 | 1 | 1 | 0 | 8.4533 | 13.6562 | 16.745 | 2.19474 | nonconstant_observed |
| theta | 0.1 | ensemble_value_disagreement_pessimism | -187.785 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 7.125 | 0.863327 | nonconstant_observed |
| theta | 0.1 | plus_adapted_ricker_only_pbvi | -94.9348 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 9.375 | 0 | constant_within_every_episode |
| theta | 0.1 | moor_adapted_ricker_misspec_pbvi | -94.9348 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 9.375 | 0 | constant_within_every_episode |
| theta | 0.2 | refplan | -187.168 | 0 | 1 | 1 | 0 | 12.834 | 20.5331 | 11.965 | 1.75145 | nonconstant_observed |
| theta | 0.2 | ogsrl | -188.194 | 0 | 1 | 1 | 0 | 39.0833 | 42.5159 | 15.4938 | 0.241815 | nonconstant_observed |
| theta | 0.2 | bamcts | -190.095 | 0 | 1 | 1 | 0 | 4.0746 | 6.72809 | 17.1125 | 2.288 | nonconstant_observed |
| theta | 0.2 | ensemble_value_disagreement_pessimism | -187.773 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 7.13438 | 0.862517 | nonconstant_observed |
| theta | 0.2 | plus_adapted_ricker_only_pbvi | -94.9348 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 9.375 | 0 | constant_within_every_episode |
| theta | 0.2 | moor_adapted_ricker_misspec_pbvi | -94.9348 | 0 | 1 | 1 | 0 | 0.523914 | 0.523914 | 9.375 | 0 | constant_within_every_episode |
| regime | 0.1 | refplan | -186.816 | 0 | 1 | 1 | 0 | 9.64218 | 19.1772 | 10.7763 | 1.77646 | nonconstant_observed |
| regime | 0.1 | ogsrl | -188.368 | 0 | 1 | 1 | 0 | 40.96 | 42.6769 | 15.9688 | 0.140471 | nonconstant_observed |
| regime | 0.1 | bamcts | -189.608 | 0 | 1 | 1 | 0 | 8.19992 | 13.2851 | 16.2088 | 2.18937 | nonconstant_observed |
| regime | 0.1 | ensemble_value_disagreement_pessimism | -187.839 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.15625 | 0.868833 | nonconstant_observed |
| regime | 0.1 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| regime | 0.1 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| regime | 0.2 | refplan | -187.239 | 0 | 1 | 1 | 0 | 11.929 | 20.2799 | 11.93 | 1.75218 | nonconstant_observed |
| regime | 0.2 | ogsrl | -188.685 | 0 | 1 | 1 | 0 | 40.9115 | 42.4505 | 16.8313 | 0.277578 | nonconstant_observed |
| regime | 0.2 | bamcts | -190.351 | 0 | 1 | 1 | 0 | 3.62191 | 6.3965 | 17.2125 | 2.29673 | nonconstant_observed |
| regime | 0.2 | ensemble_value_disagreement_pessimism | -187.818 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 7.1375 | 0.868006 | nonconstant_observed |
| regime | 0.2 | plus_adapted_ricker_only_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |
| regime | 0.2 | moor_adapted_ricker_misspec_pbvi | -94.9591 | 0 | 1 | 1 | 0 | 0.428944 | 0.428944 | 9.375 | 0 | constant_within_every_episode |

## Claim limits

- Ricker-only PLUS has a correct-form inductive bias only on Ricker truth and is
  deliberately misspecified on Allee, theta-logistic, and regime-switching truth.
- Corrected MOOR commits to one fitted Ricker model.
- Constant-policy status and action entropy must accompany return comparisons;
  exact ties can reflect policy degeneracy rather than algorithmic equivalence.
- Families are not pooled as the only headline.
