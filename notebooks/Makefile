# assumed that the ace2-paper conda environment kernal set in notebook via ipykernel
run_climate_skill:
	echo "Running ACE2-ERA5 and -SHiELD climate skill notebooks"
	jupyter nbconvert --execute --to notebook --inplace climate_skill_1deg_annual_mean_series.ipynb
	jupyter nbconvert --execute --to notebook --inplace climate_skill_1deg_zonal_time_mean.ipynb
	jupyter nbconvert --execute --to notebook --inplace climate_skill_1deg_time_mean_RMSE.ipynb
	jupyter nbconvert --execute --to notebook --inplace climate_skill_1deg_enso_correlation.ipynb

# assumed that the ace2-paper conda environment is active
run_era5:
	echo "Running ERA5 notebooks"
	jupyter nbconvert --execute --to notebook --inplace era5_daily_variability_strat.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_monthly_variability_80yr.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_tropical_cyclones.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_weather_skill.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_tropical_variability.ipynb

run_constraints:
	echo "Running constraints notebooks"
	jupyter nbconvert --execute --to notebook --inplace shield_physical_constraints_ablation.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_budget_closure.ipynb
