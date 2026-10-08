# Apify Python base image (Python 3.13 + Apify runtime conventions)
FROM apify/actor-python:3.13

USER myuser

# Dependencies first (cached layer)
COPY --chown=myuser:myuser requirements.txt ./
RUN python --version && pip install --no-cache-dir -r requirements.txt && pip freeze

COPY --chown=myuser:myuser . ./

# Catch syntax errors at build time
RUN python -m compileall -q src/

CMD ["python3", "-m", "src"]
