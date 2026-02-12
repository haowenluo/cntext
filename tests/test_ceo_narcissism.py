"""
Tests for the CEO narcissism (FPSP) pipeline.

Covers:
    1. Earnings call transcript parser (speaker diarization)
    2. Pronoun dictionary loading
    3. FPSP ratio calculation
    4. End-to-end integration with synthetic transcript
"""

import os
import math
import tempfile
import pytest
import pandas as pd

from cntext.io.earnings_call import (
    parse_earnings_call,
    get_ceo_qa_text,
    parse_earnings_call_to_df,
    extract_header_metadata,
    _identify_sections,
    _parse_participant_block,
    _is_ceo,
)
from cntext.stats.narcissism import (
    fpsp_ratio,
    fpsp_ratio_from_dict,
    _tokenize_for_pronouns,
    SINGULAR_PRONOUNS,
    PLURAL_PRONOUNS,
)

# ============================================================
# Synthetic Refinitiv StreetEvents transcript for testing
# ============================================================
SAMPLE_TRANSCRIPT = """
Refinitiv StreetEvents Event Transcript
E D I T E D   V E R S I O N

Q3 2023 Agilent Technologies Inc Earnings Call
AUGUST 15, 2023 / 8:30PM GMT

================================================================================
Corporate Participants

 * Mike McMullen Agilent Technologies Inc - President & CEO
 * Robert McMahon Agilent Technologies Inc - Senior VP & CFO
 * Parmeet Ahuja Agilent Technologies Inc - VP of IR

================================================================================
Conference Call Participants

 * Vijay Kumar Evercore ISI - Analyst
 * Rachel Vatnsdal JPMorgan - Analyst

================================================================================
Presentation

Operator [1]

Good day, ladies and gentlemen. Welcome to the Agilent Technologies third
quarter 2023 earnings conference call. Please note that this conference is
being recorded.

Mike McMullen, Agilent Technologies Inc - President & CEO [2]

Thank you, Operator. Good afternoon, everyone. I am pleased to report our
results for the third quarter. We delivered strong performance. Our team
executed well across the board.

Robert McMahon, Agilent Technologies Inc - Senior VP & CFO [3]

Thanks, Mike. I will now walk you through our financial results. We
achieved revenue of $1.67 billion. Our operating margin expanded to 28.3%.
We remain confident in our outlook.

================================================================================
Questions and Answers

Operator [1]

We will now begin the question-and-answer session. Our first question
comes from Vijay Kumar with Evercore ISI.

Vijay Kumar, Evercore ISI - Analyst [2]

Thanks for taking my question. Can you talk about the demand environment?

Mike McMullen, Agilent Technologies Inc - President & CEO [3]

Great question, Vijay. I am seeing positive trends in our end markets.
I believe my team has positioned us well. I personally drove several
key customer engagements this quarter. My view is that we are well
positioned for growth. I think I have done a good job navigating the
macro uncertainty myself.

Vijay Kumar, Evercore ISI - Analyst [4]

That is helpful. And as a follow-up, how do you think about capital
allocation?

Mike McMullen, Agilent Technologies Inc - President & CEO [5]

I am glad you asked. I think our capital allocation strategy is strong.
Me and my leadership team reviewed our priorities. I see us continuing
to invest in innovation. We believe our pipeline is robust.

Robert McMahon, Agilent Technologies Inc - Senior VP & CFO [6]

Let me add some color. We returned $250 million to shareholders. Our
balance sheet is strong. We continue to evaluate opportunities.

Rachel Vatnsdal, JPMorgan - Analyst [7]

Thanks for taking my questions. How are you thinking about M&A?

Mike McMullen, Agilent Technologies Inc - President & CEO [8]

I love that question. I am always looking for opportunities that
fit my strategic vision. I have built a strong track record with
acquisitions. My philosophy is to be disciplined. We explore
opportunities where we can add value together.

Operator [9]

This concludes our Q&A session.

Mike McMullen, Agilent Technologies Inc - President & CEO [10]

Thank you everyone for joining us today. I am excited about our future.

Operator [11]

This concludes today's call. You may now disconnect.

================================================================================
Definitions

Preliminary Transcript: An initial version of the transcript.
Edited Transcript: A revised version of the transcript.

================================================================================
Disclaimer

This transcript may contain forward-looking statements.
""".strip()


# ============================================================
# Tests: Transcript Parser
# ============================================================

class TestTranscriptParser:

    def test_identify_sections(self):
        sections = _identify_sections(SAMPLE_TRANSCRIPT)
        section_names = [s[0] for s in sections]
        assert 'Corporate Participants' in section_names
        assert 'Conference Call Participants' in section_names
        assert 'Presentation' in section_names
        assert 'Questions and Answers' in section_names

    def test_parse_corporate_participants(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        corp = parsed['corporate_participants']
        assert len(corp) == 3
        names = [p['name'] for p in corp]
        assert 'Mike McMullen' in names
        assert 'Robert McMahon' in names
        # Check title parsing
        ceo = [p for p in corp if p['name'] == 'Mike McMullen'][0]
        assert 'CEO' in ceo['title']

    def test_parse_conference_participants(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        conf = parsed['conference_participants']
        assert len(conf) == 2
        names = [p['name'] for p in conf]
        assert 'Vijay Kumar' in names
        assert 'Rachel Vatnsdal' in names

    def test_presentation_turns(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        pres = parsed['presentation']
        # Operator + Mike + Robert = 3 turns
        assert len(pres) == 3
        assert pres[0]['role'] == 'operator'
        assert pres[1]['name'] == 'Mike McMullen'
        assert pres[1]['role'] == 'corporate'
        assert pres[1]['is_ceo'] is True

    def test_qa_turns(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        qa = parsed['qa']
        # Operator[1], Vijay[2], Mike[3], Vijay[4], Mike[5],
        # Robert[6], Rachel[7], Mike[8], Operator[9], Mike[10], Operator[11]
        assert len(qa) == 11

    def test_speaker_roles(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        qa = parsed['qa']
        roles = {t['name']: t['role'] for t in qa}
        assert roles['Operator'] == 'operator'
        assert roles['Mike McMullen'] == 'corporate'
        assert roles['Robert McMahon'] == 'corporate'
        assert roles['Vijay Kumar'] == 'analyst'
        assert roles['Rachel Vatnsdal'] == 'analyst'

    def test_ceo_identification(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        ceo_turns = [t for t in parsed['qa'] if t['is_ceo']]
        # Mike speaks in turns 3, 5, 8, 10
        assert len(ceo_turns) == 4
        for t in ceo_turns:
            assert t['name'] == 'Mike McMullen'

    def test_get_ceo_qa_text(self):
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)
        ceo_text = get_ceo_qa_text(parsed)
        assert len(ceo_text) > 0
        # CEO text should contain first-person pronouns
        assert 'I am' in ceo_text or 'I believe' in ceo_text or 'i am' in ceo_text.lower()
        # Should NOT contain analyst speech
        assert 'Can you talk about the demand' not in ceo_text

    def test_parse_to_dataframe(self):
        df = parse_earnings_call_to_df(SAMPLE_TRANSCRIPT)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {'name', 'company', 'title', 'role', 'is_ceo',
                         'turn_id', 'section', 'text'}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0

    def test_extract_header_metadata(self):
        meta = extract_header_metadata(SAMPLE_TRANSCRIPT)
        assert meta['quarter'] == 'Q3 2023'
        assert 'Agilent' in meta['company_name']
        assert 'AUGUST' in meta['date']


# ============================================================
# Tests: FPSP Ratio Calculation
# ============================================================

class TestFPSPRatio:

    def test_tokenizer_preserves_I(self):
        tokens = _tokenize_for_pronouns("I believe we can do this.")
        assert 'i' in tokens
        assert 'we' in tokens

    def test_basic_ratio(self):
        # "I" = 1 singular, "we" = 1 plural -> ratio = 0.5
        text = "I believe we can do this."
        result = fpsp_ratio(text)
        assert result['singular_count'] == 1
        assert result['plural_count'] == 1
        assert result['fpsp_ratio'] == 0.5

    def test_all_singular(self):
        text = "I did this myself. My effort. Me alone."
        result = fpsp_ratio(text)
        assert result['singular_count'] > 0
        assert result['plural_count'] == 0
        assert result['fpsp_ratio'] == 1.0

    def test_all_plural(self):
        text = "We did this ourselves. Our effort. Us together."
        result = fpsp_ratio(text)
        assert result['singular_count'] == 0
        assert result['plural_count'] > 0
        assert result['fpsp_ratio'] == 0.0

    def test_no_pronouns(self):
        text = "The company reported strong results this quarter."
        result = fpsp_ratio(text)
        assert result['singular_count'] == 0
        assert result['plural_count'] == 0
        assert math.isnan(result['fpsp_ratio'])

    def test_empty_text(self):
        result = fpsp_ratio("")
        assert result['word_count'] == 0
        assert math.isnan(result['fpsp_ratio'])
        assert math.isnan(result['fpsp_per_1000'])

    def test_per_1000_words(self):
        # Build a text with known counts
        # 10 words total, 2 singular, 1 plural
        text = "I think I can help. We should try harder now."
        result = fpsp_ratio(text)
        assert result['singular_count'] == 2
        expected_per_1000 = round(2 / result['word_count'] * 1000, 4)
        assert result['fpsp_per_1000'] == expected_per_1000

    def test_return_series(self):
        text = "I believe we can do this."
        result = fpsp_ratio(text, return_series=True)
        assert isinstance(result, pd.Series)
        assert 'fpsp_ratio' in result.index

    def test_custom_pronouns(self):
        text = "I think he is great. She also agrees."
        # Custom: singular = {'i', 'he'}, plural = {'she'}
        result = fpsp_ratio(text,
                            singular={'i', 'he'},
                            plural={'she'})
        assert result['singular_count'] == 2
        assert result['plural_count'] == 1

    def test_case_insensitivity(self):
        text = "I believe My team is strong. MY vision is clear."
        result = fpsp_ratio(text)
        # "I" and "My" (x2) = 3 singular
        assert result['singular_count'] == 3

    def test_mine_and_myself(self):
        text = "This project is mine. I built it myself."
        result = fpsp_ratio(text)
        assert result['singular_count'] >= 3  # mine, I, myself

    def test_ours_and_ourselves(self):
        text = "This success is ours. We achieved it ourselves."
        result = fpsp_ratio(text)
        assert result['plural_count'] >= 3  # ours, we, ourselves


# ============================================================
# Tests: Dictionary Loading
# ============================================================

class TestPronounDictionary:

    def test_builtin_dictionary_loads(self):
        from cntext.io.dict import read_yaml_dict
        d = read_yaml_dict('en_common_Pronouns.yaml', is_builtin=True)
        assert 'Dictionary' in d
        assert 'singular' in d['Dictionary']
        assert 'plural' in d['Dictionary']
        assert 'i' in d['Dictionary']['singular']
        assert 'we' in d['Dictionary']['plural']

    def test_fpsp_ratio_from_dict(self):
        text = "I believe we can do this. My view is that our team is strong."
        result = fpsp_ratio_from_dict(text)
        assert result['singular_count'] > 0
        assert result['plural_count'] > 0
        assert 0.0 < result['fpsp_ratio'] < 1.0


# ============================================================
# Tests: CEO Title Detection
# ============================================================

class TestCEODetection:

    def test_ceo_title(self):
        assert _is_ceo('President & CEO') is True
        assert _is_ceo('Chief Executive Officer') is True
        assert _is_ceo('Chairman, President & CEO') is True

    def test_non_ceo_title(self):
        assert _is_ceo('Senior VP & CFO') is False
        assert _is_ceo('VP of IR') is False
        assert _is_ceo('Analyst') is False

    def test_case_insensitivity(self):
        assert _is_ceo('president & ceo') is True
        assert _is_ceo('CHIEF EXECUTIVE OFFICER') is True


# ============================================================
# Tests: End-to-End Integration
# ============================================================

class TestEndToEnd:

    def test_full_pipeline_on_synthetic_transcript(self):
        """Test the complete flow: parse -> isolate CEO Q&A -> compute FPSP."""
        # Step 1: Parse
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)

        # Step 2: Isolate CEO Q&A text
        ceo_text = get_ceo_qa_text(parsed)
        assert len(ceo_text) > 100  # CEO had substantial Q&A speech

        # Step 3: Compute FPSP
        result = fpsp_ratio(ceo_text)

        # The synthetic CEO speech is deliberately narcissistic
        # (heavy use of I/me/my/myself)
        assert result['singular_count'] > result['plural_count']
        assert result['fpsp_ratio'] > 0.5  # More singular than plural
        assert result['word_count'] > 0
        assert result['fpsp_per_1000'] > 0

    def test_pipeline_script_with_temp_dir(self):
        """Test the pipeline script processes a file correctly."""
        from ceo_narcissism_pipeline import process_single_transcript

        # Write the sample transcript to a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False,
                                         encoding='utf-8') as f:
            f.write(SAMPLE_TRANSCRIPT)
            tmp_path = f.name

        try:
            row = process_single_transcript(tmp_path)
            assert row is not None
            assert row['ceo_name'] == 'Mike McMullen'
            assert row['qa_fpsp_ratio'] > 0.5
            assert row['qa_word_count'] > 0
            assert row['ceo_qa_turns'] == 4
            assert 'Agilent' in row['company_name']
            assert row['quarter'] == 'Q3 2023'
        finally:
            os.unlink(tmp_path)

    def test_presentation_vs_qa_difference(self):
        """
        Verify that presentation (scripted) and Q&A (unscripted) sections
        produce different pronoun profiles.
        """
        parsed = parse_earnings_call(SAMPLE_TRANSCRIPT)

        # CEO presentation text (scripted — uses "we/our" more)
        ceo_pres = ' '.join(
            t['text'] for t in parsed['presentation'] if t['is_ceo']
        )
        # CEO Q&A text (unscripted — uses "I/my" more)
        ceo_qa = get_ceo_qa_text(parsed)

        pres_result = fpsp_ratio(ceo_pres)
        qa_result = fpsp_ratio(ceo_qa)

        # The synthetic data is designed so that Q&A has a higher FPSP ratio
        # (more narcissistic) than the presentation
        assert qa_result['fpsp_ratio'] > pres_result['fpsp_ratio']


# ============================================================
# Tests: Edge Cases
# ============================================================

class TestEdgeCases:

    def test_empty_transcript(self):
        parsed = parse_earnings_call("")
        assert parsed['presentation'] == []
        assert parsed['qa'] == []
        ceo_text = get_ceo_qa_text(parsed)
        assert ceo_text == ''

    def test_no_qa_section(self):
        # Transcript with only presentation, no Q&A delimiter
        text = """
================================================================================
Corporate Participants

 * John Doe TestCorp - CEO

================================================================================
Presentation

Operator [1]

Welcome.

John Doe, TestCorp - CEO [2]

I am pleased to report our results. We had a strong quarter.
"""
        parsed = parse_earnings_call(text)
        assert len(parsed['presentation']) == 2
        assert len(parsed['qa']) == 0
        ceo_text = get_ceo_qa_text(parsed)
        assert ceo_text == ''

    def test_transcript_with_no_ceo(self):
        text = """
================================================================================
Corporate Participants

 * Jane Smith TestCorp - CFO

================================================================================
Questions and Answers

Operator [1]

We will take questions.

Jane Smith, TestCorp - CFO [2]

I am happy to take questions. We had a great quarter.
"""
        parsed = parse_earnings_call(text)
        ceo_text = get_ceo_qa_text(parsed)
        # CFO is not CEO, so no CEO text
        assert ceo_text == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
