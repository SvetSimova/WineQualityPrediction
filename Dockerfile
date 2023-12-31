FROM python:3.11.1
WORKDIR /housepricing
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["waitress-serve", "--listen", "0.0.0.0:8080", "winequalityprediction:app"]