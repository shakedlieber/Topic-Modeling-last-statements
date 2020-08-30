import re


class Prisoner:
    def __init__(self, prisoner_id = 0, execution_date = "", state= "", method_of_execution= "", name= "", race= "", sex = "", age_at_murder = 0, age_at_execution= "", date_of_birth="", victims="", date_of_murder="", method_of_murder="", relationship_to_victims="", date_of_sentence="", last_statement="", prisoner_url=""):
        self.prisoner_id = prisoner_id
        self.execution_date = execution_date
        self.state = state
        self.method_of_execution = method_of_execution
        self.name = name
        self.race = race
        self.sex = sex
        self.age_at_murder = age_at_murder
        self.age_at_execution = age_at_execution
        self.date_of_birth = date_of_birth
        self.victims = victims
        self.date_of_murder = date_of_murder
        self.method_of_murder = method_of_murder
        self.relationship_to_victims = relationship_to_victims
        self.date_of_sentence = date_of_sentence
        self.last_statement = last_statement
        self.prisoner_url = prisoner_url

    def constructor_assist(self, prisoner_id, all_prisoner_attributes):
        self.prisoner_id = prisoner_id
        self.execution_date = all_prisoner_attributes[1].get_text().replace('\n', '').encode("ascii", "ignore")
        self.state = all_prisoner_attributes[2].get_text().replace('\n', '').encode("ascii", "ignore")
        self.method_of_execution = all_prisoner_attributes[3].get_text().replace('\n', '').encode("ascii", "ignore")
        name_with_details = all_prisoner_attributes[4].get_text().replace('\n', '')
        # name_with_details = re.split('\$', name_with_details)
        found_pattern = re.search(r'[a-zA-Z]{1}(\s)?\/(\s)?[a-zA-Z]{1}(\s)?\/(\s)?[0-9]+(\s)?(\-)?(\s)?[0-9]*',
                                  name_with_details)
        name = re.split(found_pattern.group(0), name_with_details)
        name = name[0].replace('\n', '')
        details = re.split(' / | - ', found_pattern.group(0))
        self.name = name.encode("ascii", "ignore")
        self.race = details[0].encode("ascii", "ignore")
        self.sex = details[1].encode("ascii", "ignore")
        self.age_at_murder = details[2]
        self.age_at_execution = -1
        if len(details) == 4:
            self.age_at_execution = details[3]
        self.date_of_birth = all_prisoner_attributes[5].get_text().replace('\n', '').encode("ascii", "ignore")
        self.victims = all_prisoner_attributes[6].get_text().replace('$', ' ').replace('\n', '').encode("ascii",
                                                                                                        "ignore")
        self.date_of_murder = all_prisoner_attributes[7].get_text().replace('\n', '').encode("ascii", "ignore")
        self.method_of_murder = all_prisoner_attributes[8].get_text().replace('\n', '').encode("ascii", "ignore")
        self.relationship_to_victims = all_prisoner_attributes[9].get_text().replace('\n', '').encode("ascii",
                                                                                                      "ignore")
        self.date_of_sentence = all_prisoner_attributes[10].get_text().replace('\n', '').encode("ascii", "ignore")
        self.last_statement = ''

    def set_last_statement(self, last_statement):
        self.last_statement = last_statement

    def set_prisoner_url(self, prisoner_url):
        self.prisoner_url = prisoner_url

    def __iter__(self):
        return iter([
            self.prisoner_id,
            self.execution_date,
            self.state,
            self.method_of_execution,
            self.name,
            self.race,
            self.sex,
            self.age_at_murder,
            self.age_at_execution,
            self.date_of_birth,
            self.victims,
            self.date_of_murder,
            self.method_of_murder,
            self.relationship_to_victims,
            self.date_of_sentence,
            self.last_statement,
            self.prisoner_url])
