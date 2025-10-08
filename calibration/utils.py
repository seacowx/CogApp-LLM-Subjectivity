DEMOGRAPHIC_TEMPLATE = lambda age, gender, education, ethnicity: f'''
You are a {age} years old {ethnicity.lower()} {gender.lower()} whose education level is "{education}".
'''

PERSONALITY_TRAITS_TEMPLATE = lambda openness, conscientiousness, extraversion, agreeableness, emotional_stability: f'''
You are a {openness}{conscientiousness}{extraversion}{agreeableness}{emotional_stability} person.
'''


def make_demographic_info(selected_demographic_info: dict) -> str:
    return DEMOGRAPHIC_TEMPLATE(
        age=selected_demographic_info['age'],
        gender=selected_demographic_info['gender'],
        education=selected_demographic_info['education'],
        ethnicity=selected_demographic_info['ethnicity'],
    ).strip()


def make_traits_info(selected_demographic_info: dict) -> str:

    openness_to_experience = '' if \
        selected_demographic_info['open'] == selected_demographic_info['conventional'] else \
            'open, ' if selected_demographic_info['open'] > selected_demographic_info['conventional'] \
            else 'conventional, '

    conscientiousness = '' if \
        selected_demographic_info['dependable'] == selected_demographic_info['disorganized'] else \
            'dependable, ' if selected_demographic_info['dependable'] > selected_demographic_info['disorganized'] \
            else 'disorganized, '

    extraversion = '' if \
        selected_demographic_info['extravert'] == selected_demographic_info['quiet'] else \
            'extravert, ' if selected_demographic_info['extravert'] > selected_demographic_info['quiet'] \
            else 'quiet, '

    agreeableness = '' if \
        selected_demographic_info['sympathetic'] == selected_demographic_info['critical'] else \
            'sympathetic, ' if selected_demographic_info['sympathetic'] > selected_demographic_info['critical'] \
            else 'critical, '

    emotional_stability = '' if \
        selected_demographic_info['anxious'] == selected_demographic_info['calm'] else \
            'anxious, ' if selected_demographic_info['anxious'] > selected_demographic_info['calm'] \
            else 'calm, '

    cur_personality_traits = [
        openness_to_experience,
        conscientiousness,
        extraversion,
        agreeableness,
        emotional_stability,
    ]

    if any([ele != '' for ele in cur_personality_traits]):
        cur_personality_traits = PERSONALITY_TRAITS_TEMPLATE(
            openness=openness_to_experience,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            emotional_stability=emotional_stability,
        )
    else:
        cur_personality_traits = ''

    cur_personality_traits = cur_personality_traits.rsplit(', ', 1)[0]
    if cur_personality_traits.count(', ') >= 1:
        cur_personality_traits, last_trait = cur_personality_traits.rsplit(', ', 1)
        if cur_personality_traits.count(', ') == 1:
            cur_personality_traits += f' and {last_trait} person.'
        else:
            cur_personality_traits += f', and {last_trait} person.'
    else:
        cur_personality_traits += ' person.'

    return cur_personality_traits.strip()


def sanity_check(parsed_args):
    if parsed_args.eval_method not in ['pair-rank', 'avg-conf', 'consistency']:
        raise ValueError(f'Invalid evaluation method: {parsed_args.eval_method}')

