from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/record', methods=['POST'])
def new_register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    required_fields = ["traffic_cam_id", "start_datetime", "end_datetime", "vehicle_count", "average_speed"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    print(f"Received new register: {data}")

    # TODO: Save data to database

    return jsonify({"message": "Register received successfully"}), 200

if __name__ == '__main__':
    app.run(host="localhost", port=6000, debug=True)
