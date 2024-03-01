import yaml
import pandas as pd


def update_concept_targets(ems_ontology_df):
    # FIRST: read the existing yml file into a map
    # map relates strings -> sets
    dict_concepts_to_sets = {}
    dict_metadata = {}
    with open(r'data/pycontextnlp_targets.yml') as file:
        concepts_list = yaml.load_all(file, Loader=yaml.FullLoader)
        for concept in concepts_list:
            concept_to_add = ''

            for k, v in concept.items():
                if k == 'Lex':
                    concept_to_add = v

                if k == 'Regex':
                    if v == '':
                        dict_concepts_to_sets[concept_to_add] = set()
                    else:
                        first_last_chars_removed = v[1:-1]
                        list_regexes = first_last_chars_removed.split('|')
                        dict_concepts_to_sets[concept_to_add] = set(list_regexes)

            dict_metadata[concept_to_add] = {}

            for k, v in concept.items():
                if k != 'Regex' and k != 'Lex':
                    dict_metadata[concept_to_add][k] = v


    # SECOND: read the ems ontology df
    # if concept exists, add the regex to that set
    for index, row in ems_ontology_df.iterrows():
        row_ontology_concept = row[0]
        row_ontology_regex = row[1]

        if row_ontology_concept in dict_concepts_to_sets:

            is_substring = False
            for concept, set_regex in dict_concepts_to_sets.items():
                if is_substring:
                    break
                for regex in set_regex:
                    if row_ontology_regex in regex or regex in row_ontology_regex:
                        is_substring = True
                        break

            if not is_substring:
                dict_concepts_to_sets[row_ontology_concept].add(row_ontology_regex)

            # # used for testing
            # dict_concepts_to_sets[row_ontology_concept].add(row_ontology_regex)

        # # if concept in ontology does not exist in our dictionary, add it with a new set
        # else:
        #     dict_concepts_to_sets[row_ontology_concept] = set()
        #     dict_metadata[row_ontology_concept] = {}
        #     dict_metadata[row_ontology_concept]['Comments'] = ''
        #     dict_metadata[row_ontology_concept]['Direction'] = ''
        #     dict_metadata[row_ontology_concept]['Type'] = row_ontology_concept.upper().replace(' ', '_')


    # THIRD: convert our map of concept -> regex back to a yaml file
    # dump_all() expects a list of dictionaries, each dictionary should be an individual document
    list_dicts = []
    for concept in dict_concepts_to_sets:
        str_regex_statement = ''
        if(len(dict_concepts_to_sets[concept]) != 0):
            str_regex_statement = '('
            for regex_to_concat in dict_concepts_to_sets[concept]:

                # str_regex_statement += '\s+' + regex_to_concat + '\.?\!?\??' + '\,?\:?\-?' + '\s+' + '|'
                str_regex_statement += '(\s|\/)' + regex_to_concat + '(\s|\/|\.|\!|\?|\,|\:|\-)' + '|'

            str_regex_statement = str_regex_statement[:-1]
            str_regex_statement += ')'
        dict_to_add = {}
        dict_to_add['Lex'] = concept
        dict_to_add['Regex'] = str_regex_statement
        # add metadata back to dict_to_add
        for metadata_piece in dict_metadata[concept]:
            dict_to_add[metadata_piece] = dict_metadata[concept][metadata_piece]
        list_dicts.append(dict_to_add)

    with open(r'data/pycontextnlp_targets.yml', 'w') as file:
        documents = yaml.dump_all(list_dicts, file, default_flow_style=False)


def main():
    ems_ontology_df = pd.read_excel("data/All_In_One_Extensions_ProtocolModeling_Weighted.xls")
    update_concept_targets(ems_ontology_df)


if __name__ == "__main__":
    main()