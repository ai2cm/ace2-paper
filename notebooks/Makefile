
# assumed that the ace2-paper conda environment is active
run_era5:
	echo "Running ERA5 notebooks"
	jupyter nbconvert --execute --to notebook --inplace era5_daily_variability_strat.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_monthly_variability_80yr.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_tropical_cyclones.ipynb
	jupyter nbconvert --execute --to notebook --inplace era5_weather_skill.ipynb
