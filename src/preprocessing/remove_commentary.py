import pandas as pd

def extract_parentheses(text: str, parenthese_chars: str = "()") -> list[str]:
    # Returns every piece of text inside parenthesis, if we are dealing with nested parenthesis we won't consider inner substrings only outermost one.
    stack = []
    result = []
    for idx, ch in enumerate(text):
        if ch == parenthese_chars[0]:
            stack.append(idx)
        elif ch == parenthese_chars[1]:
            if stack:
                start = stack.pop()

                if not stack:  # we are only taking outermost parentheses hence check for emptiness
                    result.append(text[start + 1: idx])
    return result



def extract_commentary(df: pd.DataFrame, text_column: str = 'text') -> pd.Series:
    inside_parenthesis = df[text_column].apply(lambda text: extract_parentheses(text))
    inside_brackets = df[text_column].apply(lambda text: extract_parentheses(text, parenthese_chars="[]"))
    combined = pd.concat([inside_parenthesis, inside_brackets])
    combined = combined.explode()  # One comment may include several bracketed text so we need to flatten our entries
    return combined[~combined.isna()]



def identify_removable_parts(commentary) -> list[pd.Series]:
    # Most straightforward, take all comments that include speaker
    print("Found", len(commentary), "bracketed comments")
    speaker_commentary = commentary[commentary.str.lower().str.contains('speaker|mep')]

    # considering rest
    non_speaker_commentary = commentary[~commentary.str.lower().str.contains('speaker')]

    microphone_commentary = non_speaker_commentary[(non_speaker_commentary.str.lower().str.contains('microphone|mic')) ]

    # This is all the unique bracketed texts, based on my brief scan of these texts I haven't seen anything we can discard
    leave_in = non_speaker_commentary.value_counts()[non_speaker_commentary.value_counts() == 1].index
    non_speaker_commentary = non_speaker_commentary[~non_speaker_commentary.apply(lambda text: text in leave_in)]

    # text inside the parenthesis which is part of the real debate is probably longer, I don't see any comments that might be important to infer the context
    short_non_speaker_commentary = non_speaker_commentary[non_speaker_commentary.str.split().str.len() <= 2]

    
    long_non_speaker_commentary = non_speaker_commentary[non_speaker_commentary.str.split().str.len() > 2]
    lowercase_longer_comments = non_speaker_commentary.str.lower()
    contained_patterns = long_non_speaker_commentary[long_non_speaker_commentary.str.contains('applause|rule 1|speaking session')]
    repeating_commentary_beginnings = long_non_speaker_commentary[long_non_speaker_commentary.str.startswith(('parliament', 'the parliament', 'the sitting', 'the mep', 'the president',
                                                                                                     'end of', 'the oral amendment',
                                                                                                     'explanation of vote', 'article', 'rule'))]
    # There are still 2.9K unique comments left all of them are infrequent and also seemed relevant for the context
    # I couldn't find any other patterns
    n_residual = len(non_speaker_commentary) - len(microphone_commentary) - len(short_non_speaker_commentary) - len(contained_patterns) - len(repeating_commentary_beginnings)
    print("Leaving",n_residual, "bracketed comments")
    return [speaker_commentary, microphone_commentary, short_non_speaker_commentary, 
             repeating_commentary_beginnings, contained_patterns]


def remove_from_text(original: str, strings_to_remove: list[str] ) -> str:
    if not isinstance(strings_to_remove, list):  # skipping over NaN value
        return original
    for to_remove in strings_to_remove:
        # Durning commentary extraction we are only extracting text within parentheses, so here I'm adding it back to get rid of them as well when removing commentary from the text
        original = original.replace(f"({to_remove})", "")
        original = original.replace(f"[{to_remove}]", "")

    return original.strip()


def remove_commentary(df: pd.DataFrame, text_column: str = "translatedText") -> pd.DataFrame:
    commentary = extract_commentary(df, text_column=text_column)
    commentary = pd.concat(identify_removable_parts(commentary))
    
    commentary.name = 'commentary'
    commentary = commentary.groupby(commentary.index).agg(lambda comments: list(comments))

    merged = pd.merge(df, commentary, left_index=True, right_index=True, how='left')
    merged[text_column] = merged[[text_column, 'commentary']].apply(lambda row: remove_from_text(row['translatedText'], row['commentary']), axis=1)
    return merged.drop('commentary', axis=1)



    
# some more pattern suggestions:
# "the MEP"
# "the parliament decides"
# "the sitting is | was"