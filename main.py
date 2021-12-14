
import json
import nbimporter
import sys
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import spacy



# load english language model
nlp = spacy.load("en_core_web_sm")

animal_cell_filename = 'animal_cell_span_to_question.jsonl'

igneous_rock_v1_qasrl_result_filename = 'igneous_rock_span_to_question_v1_coref.jsonl'
igneous_rock_v2_qasrl_result_filename = 'igneous_rock_span_to_question_v2_coref.jsonl'

factory_qasrl_result_filename = 'factory_span_to_question_coref.jsonl'
animal_cell_qasrl_result_filename = 'animal_cell_span_to_question_coref.jsonl'


animal_cell_coref_filename = 'animal_cell_coref_text.txt'

factory_filename = 'factory_span_to_question.jsonl'
factory_coref_filename = 'factory_coref.jsonl'

igneous_rock_v1_filename = 'igneous_rock_span_to_question_v1.jsonl'
igneous_rock_v2_filename = 'igneous_rock_span_to_question_v2.jsonl'

animal_cell_text_filename = 'animal_cell.txt'

animal_cell_coref_to_qasrl_filename = "animal_cell_coref_to_qasrl.txt"

possible_questions = {'what', 'who', 'which', 'where'}

def main():

    # verbs_igneous_rock_v1 = read_parsed_qasrl(igneous_rock_v1_filename)
    # verbs_igneous_rock_v2 = read_parsed_qasrl(igneous_rock_v2_filename)
    # print(verbs_igneous_rock_v1)
    # print(verbs_igneous_rock_v2)

    # coref_table = read_coref(animal_cell_coref_filename)
    # verbs_animal_cell = read_parsed_qasrl(animal_cell_filename)
    # verbs_factory = read_parsed_qasrl(factory_filename)
    # print(verbs_animal_cell)
    # print(verbs_factory)

    # write_input_to_qasrl(animal_cell_coref_to_qasrl_filename, animal_cell_coref_filename)

    animal_cell = read_parsed_qasrl(animal_cell_qasrl_result_filename)
    # factory = read_parsed_qasrl(factory_qasrl_result_filename)
    # v1 = read_parsed_qasrl(igneous_rock_v1_qasrl_result_filename)
    # v2 = read_parsed_qasrl(igneous_rock_v2_qasrl_result_filename)

def get_sent_words_pos(sentence_tokens):
    sentence_tokens_str = " ".join(sentence_tokens[:-1])
    doc = nlp(sentence_tokens_str)
    sentence = next(doc.sents)
    sentence_words_pos = {}
    for token in sentence:
        if token not in sentence_words_pos:
            sentence_words_pos[str(token)] = token.pos_
    return sentence_words_pos

def sentence_starts_with_verb(sentence_tokens, sentence_verbs_indices, entity):
    verbs = [sentence_tokens[idx] for idx in sentence_verbs_indices]
    sentence_words_pos = get_sent_words_pos(sentence_tokens)
    has_seen_noun_or_propn = False
    for word in entity.split(' '):
        if word in verbs and not has_seen_noun_or_propn:
            return True
        if sentence_words_pos[word] in ['PROPN', 'NOUN']:
            has_seen_noun_or_propn = True
            continue
    return False


def write_input_to_qasrl(output_filename, input_filename):
    inp = open(input_filename, 'r')
    out = open(output_filename, 'w')
    lines = inp.readlines()
    for i, line in enumerate(lines):
        out.write(str(i+1) + "\t" + line)




def read_animal_cell_text(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    res = []
    for line in lines:
        line_list = line.split()
        for word in line_list:
            res.append(word)
    return res

def read_coref(filename):
    coref_table = {}
    f = open(filename, "r")
    lines = f.readlines()
    for idx, line in enumerate(lines):
        json_object = json.loads(line)
        for key, item in json_object.items():
            coref_table[key] = item
    return coref_table


def get_template_from_q_slots(q_slots):
    template = []
    for key, val in q_slots.items():
        if key == 'wh':
            template.append(val)
        elif key == 'subj' and val != '_':
            template.append(val)
        elif key == 'verb' and val != '_':
            template.append('<verb>')
        elif key == 'obj' and val != '_':
            template.append(val)
        elif key == 'prep' and val != '_':
            template.append(val)
        elif key == 'obj2' and val != '_':
            template.append(val)
    return " ".join(template) + "?"


def read_parsed_qasrl(filename):
    f = open(filename, "r")
    lines = f.readlines()
    question_answers_map = {}
    connected_questions = {}
    for line_idx, line in enumerate(lines):

        json_object = json.loads(line)
        sentence_id = json_object['sentenceId']
        sentence_tokens = json_object['sentenceTokens']
        print("sentence " + sentence_id + ": " + " ".join(sentence_tokens))

        sentence_verbs_indices = [d['verbIndex'] for d in json_object['verbs']]
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


            for beam_before in beams_before_verb:
                question = beam_before['questions'][0]
                if question['questionSlots']['wh'] not in possible_questions:
                    continue
                q_slots, q, verb_time = get_question_from_questions_slots(question['questionSlots'], verb)
                q_verb = '<verb>' if q_slots['verb'] != '_' else '_'
                q_sub_verb_obj = q_slots['subj'] + q_verb + q_slots['obj']
                entity_before = " ".join(sentence_tokens[beam_before['span'][0]:beam_before['span'][1]])
                print("Entity before: " + entity_before)

                ans_prob = round(beam_before['spanProb'], 2)
                q_prob = round(question['questionProb'], 2)
                print("question with answer before verb: " + q + "\nanswer: " + entity_before + ", answer prob:" + str(ans_prob) + ", question_prob:" + str(q_prob))
                if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
                    print("Filter out QA because this question contains subj+verb+obj: " + entity_before + ", " + q)
                    continue
                if sentence_starts_with_verb(sentence_tokens, sentence_verbs_indices, entity_before):
                    print("Filter out QA because answer starts with a verb: " + entity_before + ", " + q)
                    continue
                print()

                if (q, q_sub_verb_obj) in question_answers_map:
                    question_answers_map[(q, q_sub_verb_obj)].append(entity_before)
                else:
                    question_answers_map[(q, q_sub_verb_obj)] = [entity_before]

                connected_questions[(q, q_sub_verb_obj)] = []

            for beam_after in beams_after_verb:
                question = beam_after['questions'][0]
                if question['questionSlots']['wh'] not in possible_questions:
                    continue
                q_slots, q, changed_from_passive_to_active = get_question_from_questions_slots(question['questionSlots'], verb)
                if q_slots['obj2'] != '_':
                    print(1)
                q_sub_verb_obj = q_slots['subj'] + q_verb + q_slots['obj']
                entity_after = " ".join(sentence_tokens[beam_after['span'][0]:beam_after['span'][1]])
                print("Entity after: " + entity_after)

                ans_prob = round(beam_after['spanProb'], 2)
                q_prob = round(question['questionProb'], 2)
                print("question with answer after verb: " + q + "\nanswer: " + entity_after + ", answer prob:" + str(
                    ans_prob) + ", question_prob:" + str(q_prob))
                if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
                    print("Filter out QA because this question contains subj+verb+obj: " + entity_after + ", " + q)
                    continue
                if sentence_starts_with_verb(sentence_tokens, sentence_verbs_indices, entity_after):
                    print("Filter out because answer starts with a verb: " + entity_after + ", " + q)
                    continue
                print()



                if (q, q_sub_verb_obj) in question_answers_map:
                    question_answers_map[(q, q_sub_verb_obj)].append(entity_after)
                else:
                    question_answers_map[(q, q_sub_verb_obj)] = [entity_after]

        print("\n")
    print("Answers: ")
    for key, val in question_answers_map.items():
        for item in val:
            print("'" + str(item) + "',")
    print()
    answer_question_map = {}
    for key, val in question_answers_map.items():
        for item in val:
            answer_question_map[item] = key

    print("Answer Question Map")
    print("{")
    for key, val in answer_question_map.items():
        print("'" + key + "':", val, ',')
    print("}")

    for i in range(10):
        print("----------------------")



def get_active_q_slots_from_passive(question_slots, verb):
    question_slots['aux'] = "_"
    question_slots['verb'] = 'presentSingular3rd'
    question_slots['obj'] = 'something'
    question_slots['prep'] = "_"
    question_slots['obj2'] = "_"

    question = question_slots['wh'] + " " + verb['verbInflectedForms'][question_slots['verb']] + " " + question_slots['obj'] + "?"
    return question_slots, question, True

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
                        if word in verb['verbInflectedForms'] and val_list[0] in ['be', 'being']:
                            verb_time = word
                            verb_to_append.append(verb['verbInflectedForms'][word])
                        else:
                            verb_to_append.append(word)
                    result.append(" ".join(verb_to_append))
                else:
                    result.append(verb['verbInflectedForms'][val])
            else:
                result.append(val)

    return question_slots, " ".join(result) + "?", verb_time



if __name__ == '__main__':
    main()

