profile-all:
	@echo "Profiling all modules in src/..."
	@find src -type f -name "*.py" ! -name "__init__.py" | \
	while read file; do \
		mod=$$(echo "$$file" | sed 's|/|.|g; s|\.py$$||'); \
		echo "Profiling module: $$mod"; \
		poetry run pyinstrument -m $$mod --html -o pyinstrument_$$(echo $$mod | tr '.' '_').html || true; \
	done
