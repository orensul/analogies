
import json

import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
animal_cell_filename = 'animal_cell_span_to_question.jsonl'
factory_filename = 'factory_span_to_question.jsonl'
igneous_rock_v1_filename = 'igneous_rock_span_to_question_v1.jsonl'
igneous_rock_v2_filename = 'igneous_rock_span_to_question_v2.jsonl'


def main():
    # verbs_igneous_rock_v1 = read_parsed_qasrl(igneous_rock_v1_filename)
    # verbs_igneous_rock_v2 = read_parsed_qasrl(igneous_rock_v2_filename)
    # print(verbs_igneous_rock_v1)
    # print(verbs_igneous_rock_v2)

    verbs_animal_cell = read_parsed_qasrl(animal_cell_filename)
    verbs_factory = read_parsed_qasrl(factory_filename)
    print(verbs_animal_cell)
    print(verbs_factory)





def get_synonyms(word):
    synonyms = []
    verb = "create"
    for syn in wn.synsets(verb):
        if syn._pos != 'v':
            continue
        for l in syn.lemmas():
            if '_' in l.name() or l.name() == verb:
                continue
            synonyms.append(l.name())

    return set(synonyms)


def read_parsed_qasrl(filename):
    f = open(filename, "r")
    lines = f.readlines()
    verbs = []
    for line_idx, line in enumerate(lines):

        json_object = json.loads(line)
        sentence_id = json_object['sentenceId']
        sentence_tokens = json_object['sentenceTokens']
        print("sentence " + sentence_id + ": " + " ".join(sentence_tokens))
        for idx, verb in enumerate(json_object['verbs']):
            verb_idx = verb['verbIndex']
            print("verb " + str(idx+1) + " (original): " + sentence_tokens[verb_idx])
            print("verb " + str(idx+1) + " (stem): " + verb['verbInflectedForms']['stem'])
            beams_before_verb = []
            beams_after_verb = []
            for beam in verb['beam']:
                span_start = beam['span'][0]
                span_end = beam['span'][1]
                if span_end <= verb_idx:
                    beams_before_verb.append(beam)
                elif span_start > verb_idx:
                    beams_after_verb.append(beam)

            entity_before, entity_after = "", ""
            if len(beams_before_verb) > 0:
                beam_before_idx = get_max_span_prob_beam_idx(beams_before_verb)
                beam_before_verb = beams_before_verb[beam_before_idx]
                question_beam_before = beam_before_verb['questions'][0]
                question, verb_time = get_question_from_questions_slots(question_beam_before['questionSlots'], verb)
                entity_before = " ".join(sentence_tokens[beam_before_verb['span'][0]:beam_before_verb['span'][1]])
                print("question with answer before verb: " + question + "\nanswer: " + entity_before)

            if len(beams_after_verb) > 0:
                beam_after_idx = get_max_span_prob_beam_idx(beams_after_verb)
                beam_after_verb = beams_after_verb[beam_after_idx]
                question_beam_after = beam_after_verb['questions'][0]
                question, verb_time = get_question_from_questions_slots(question_beam_after['questionSlots'], verb)
                entity_after = " ".join(sentence_tokens[beam_after_verb['span'][0]:beam_after_verb['span'][1]])
                print("question with answer after verb: " + question + "\nanswer: " + entity_after)

            if verb_time in ['past', 'pastParticiple']:
                entity_before, entity_after = entity_after, entity_before
            verbs.append({"entities": (entity_before, entity_after), "verb": verb['verbInflectedForms']['stem'], "sentence_id": line_idx+1})

        print("\n")
    return verbs


def get_question_from_questions_slots(question_slots, verb):
    result = []
    verb_time = None
    for key, val in question_slots.items():
        if val != '_':
            if key == 'verb':
                val_list = val.split(' ')
                if len(val_list) > 1:
                    verb_to_append = []
                    for word in val_list:
                        if word in verb['verbInflectedForms'] and val_list[0] == 'be':
                            verb_time = word
                            verb_to_append.append(verb['verbInflectedForms'][word])
                        else:
                            verb_to_append.append(word)
                    result.append(" ".join(verb_to_append))
                else:
                    result.append(verb['verbInflectedForms'][val])
            else:
                result.append(val)
    return " ".join(result) + "?", verb_time

def get_max_span_prob_beam_idx(beams):
    max_span_prob = 0
    max_span_prob_idx = 0
    for idx, beam in enumerate(beams):
        if beam['spanProb'] > max_span_prob:
            max_span_prob = beam['spanProb']
            max_span_prob_idx = idx
    return max_span_prob_idx


if __name__ == '__main__':
    main()

