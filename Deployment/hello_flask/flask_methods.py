from flask import Flask, jsonify, request

app = Flask(__name__)


marks = [
    {"name": "Rahul", "maths": "88"},
    {"name": "Sachin", "maths": "80"},
    {"name": "Virat", "maths": "96"},
    {"name": "Rohit", "maths": "89"},
]


@app.route("/", methods=["GET"])
def hello_world():
    return jsonify({"message": "Hello, World!"})


@app.route("/marks", methods=["GET"])
def marks_all():
    return jsonify({"marks": marks})


@app.route("/marks/<string:name>", methods=["GET"])
def marks_one(name):
    mark_one = marks[0]
    for i, q in enumerate(marks):
        if q["name"] == name:
            mark_one = marks[i]
    return jsonify({"marks": mark_one})


@app.route("/marks", methods=["POST"])
def marks_add():
    new_marks = request.get_json(force=True)
    marks.append(new_marks)
    return jsonify({"marks": marks})


@app.route("/marks/<string:name>", methods=["PUT"])
def marks_edit(name):
    new_mark = request.get_json(force=True)
    for i, q in enumerate(marks):
        if q["name"] == name:
            marks[i] = new_mark
    qs = request.get_json()
    return jsonify({"marks": marks})


@app.route("/marks/<string:name>", methods=["DELETE"])
def marks_delete(name):
    for i, q in enumerate(marks):
        if q["name"] == name:
            del marks[i]
    return jsonify({"marks": marks})


if __name__ == "__main__":
    app.run()
