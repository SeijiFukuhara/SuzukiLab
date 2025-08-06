        temp_flow_path = "temp_uploaded.csv"
        with open(temp_flow_path, "wb") as f:
            f.write(fname_flow.read())