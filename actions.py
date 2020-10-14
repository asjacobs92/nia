def build_waypoint_nip(req):
    result = req.get("result")
    parameters = result.get("parameters")

    middleboxes = parameters.get("middlebox")
    target = parameters.get("policy-target")
    print("args", middleboxes, target)
    nip = ("define intent customIntent:" +
           "\n   add {}".format(''.join(map(lambda mb: "middlebox(" + mb + "), ", middleboxes))) +
           "\n   for {}".format(''.join(map(lambda pt: "client(" + pt + "), ", target))))

    speech = "The info you gave me generated this program:\n " + nip + "\n Is this what you want?"

    print("Response:", speech)

    return {
        "speech": speech,
        "displayText": speech,
        # "data": data,
        # "contextOut": [],
        "source": "nia"
    }


actions = {
    "input.waypoint": build_waypoint_nip
}
