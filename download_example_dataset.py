import os
import urllib.request
import zipfile


def download_with_url(
    url_string, download_file_path="hpa_dataset_interactiveML.zip", unzip=True
):
    with urllib.request.urlopen(url_string) as response, open(
        download_file_path, "wb"
    ) as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(download_file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(download_file_path))


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    dataset_path = "./data/hpa_dataset_v2.zip"
    if not os.path.exists(dataset_path):
        url = "https://kth.box.com/shared/static/hcnspau5lndyhkkzgv2ygsyq1978qo90.zip"
        print("downloading dataset from " + url)
        download_with_url(url, dataset_path, unzip=True)
        print("dataset saved to " + dataset_path)
    else:
        print(
            "dataset already download, if you want to download again, please remove "
            + dataset_path
        )
