import re


class ClinicalParser:

    def parse(self, text: str):

        data = {
            "age": None,
            "sex": None,
            "drug": [],
            "dose": [],
            "duration": [],
            "side_effects": [],
        }

        age = re.search(r"\b([1-9][0-9])\s*(years?|yo|y/o)\b", text, re.I)
        if age:
            data["age"] = int(age.group(1))

        text_lower = text.lower()

        if "male" in text_lower:
            data["sex"] = "male"

        elif "female" in text_lower:
            data["sex"] = "female"

        drugs = [
            "finasteride",
            "dutasteride",
            "minoxidil"
        ]

        for drug in drugs:
            if drug in text_lower:
                data["drug"].append(drug)

        doses = re.findall(r"\b\d+(\.\d+)?\s*(mg|ml|%)", text, re.I)

        for dose in doses:
            data["dose"].append("".join(dose))

        return data