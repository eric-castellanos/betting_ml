profile-all:
	@echo "Profiling functions in all modules"
	@poetry run python src.utils.profile_utils.py