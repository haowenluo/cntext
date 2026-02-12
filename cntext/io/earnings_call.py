"""
Earnings call transcript parser for Refinitiv StreetEvents format.

Parses structured earnings call transcripts into speaker-level turns,
with section identification (Presentation vs Q&A) and speaker role
classification (Corporate vs Conference Call / Analyst).

This module supports the CEO narcissism (FPSP) research pipeline by
enabling isolation of specific speakers' unscripted speech in Q&A sections.

References:
    Aktas, N., de Bodt, E., Bollaert, H., & Roll, R. (2016).
        CEO narcissism and the takeover process. Journal of Financial
        and Quantitative Analysis, 51(1), 113-137.
    Capalbo, F., Frino, A., Lim, M. Y., Mollica, V., & Palumbo, R. (2018).
        CEO narcissism and earnings management. Working Paper.
"""

import re
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ============================================================
# Section delimiter pattern used in Refinitiv StreetEvents
# ============================================================
SECTION_DELIMITER = re.compile(r'^={10,}', re.MULTILINE)

# Speaker header pattern:  Name, Company - Title [turn_number]
# Also handles "Operator [N]"
SPEAKER_HEADER = re.compile(
    r'^[-*\s]*'                          # optional leading bullets / whitespace
    r'(?P<speaker_line>.+?)'             # speaker line content (greedy-minimal)
    r'\s*\[(?P<turn_id>\d+)\]\s*$',      # [N] at end of line
    re.MULTILINE
)

# Pattern to separate name from company-title on the speaker line
# e.g. "Mike McMullen, Agilent Technologies Inc - President & CEO"
SPEAKER_DETAIL = re.compile(
    r'^(?P<name>.+?)\s*[,;]\s*(?P<company>.+?)\s*-\s*(?P<title>.+)$'
)

# Operator pattern (no company/title)
OPERATOR_PATTERN = re.compile(r'^Operator$', re.IGNORECASE)

# Known CEO title keywords (case-insensitive matching)
CEO_TITLE_KEYWORDS = [
    'chief executive officer',
    'ceo',
    'president & ceo',
    'president and ceo',
    'president and chief executive officer',
    'president & chief executive officer',
    'chairman & ceo',
    'chairman and ceo',
    'chairman & chief executive officer',
    'chairman and chief executive officer',
    'chairman, president & ceo',
    'chairman, president and ceo',
]


def _identify_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split a Refinitiv transcript into named sections.

    Returns:
        List of (section_name, section_body) tuples.
        Common section names: 'header', 'Corporate Participants',
        'Conference Call Participants', 'Presentation',
        'Questions and Answers', 'Definitions', 'Disclaimer'.
    """
    parts = SECTION_DELIMITER.split(text)
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # The first non-empty line of each section is typically its heading
        lines = part.split('\n', 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ''

        # Normalise known heading variants
        heading_lower = heading.lower()
        if 'corporate participant' in heading_lower:
            heading = 'Corporate Participants'
        elif 'conference call participant' in heading_lower:
            heading = 'Conference Call Participants'
        elif heading_lower.startswith('presentation'):
            heading = 'Presentation'
        elif 'question' in heading_lower and 'answer' in heading_lower:
            heading = 'Questions and Answers'
        elif 'definition' in heading_lower:
            heading = 'Definitions'
        elif 'disclaimer' in heading_lower:
            heading = 'Disclaimer'

        sections.append((heading, body))
    return sections


def _parse_participant_block(block: str) -> List[Dict[str, str]]:
    """
    Parse a 'Corporate Participants' or 'Conference Call Participants' block.

    Handles two common formats found in Refinitiv transcripts:
        * Name, Company - Title          (comma-separated)
        * Name  Company - Title          (space-separated, no comma)

    Returns:
        List of dicts with keys: name, company, title
    """
    # Pattern for "Name Company - Title" (no comma between name and company).
    # Strategy: split on " - " to get (name_and_company, title), then find
    # the boundary between name and company.  Company names typically contain
    # Inc, Corp, Ltd, LLC, LP, Group, Holdings, Technologies, etc., or are
    # multi-word capitalised sequences.  We look for the longest right-side
    # substring that looks like a company name.
    COMPANY_SUFFIX_RE = re.compile(
        r'\b(?:Inc|Corp|Corporation|Ltd|LLC|LP|LLP|Group|Holdings|'
        r'Technologies|Therapeutics|Pharmaceuticals|Partners|Capital|'
        r'Securities|Management|Advisors|Research|Associates|Financial|'
        r'Bank|Bancorp|Insurance|Co|Company|PLC|SA|AG|NV|SE|GmbH|'
        r'Enterprises|International|Global|Systems|Solutions|Services|'
        r'Networks|Communications|Industries|Funds?)\b',
        re.IGNORECASE,
    )

    participants = []
    for line in block.split('\n'):
        line = line.strip().lstrip('*').strip()
        if not line:
            continue

        # Try comma-separated format first: "Name, Company - Title"
        m = SPEAKER_DETAIL.match(line)
        if m:
            participants.append({
                'name': m.group('name').strip(),
                'company': m.group('company').strip(),
                'title': m.group('title').strip(),
            })
            continue

        # Try space-separated format: "Name Company - Title"
        if ' - ' in line:
            left, title = line.rsplit(' - ', 1)
            left = left.strip()
            title = title.strip()

            # Find company name boundary using company-suffix keywords
            match = COMPANY_SUFFIX_RE.search(left)
            if match:
                # Walk backwards from the suffix keyword to find where
                # the company name starts. Company names are typically
                # consecutive capitalised words.
                suffix_start = match.start()
                words = left[:suffix_start].rstrip().split()
                # Scan backwards: include capitalised words as part of company
                company_start_idx = len(words)
                for i in range(len(words) - 1, -1, -1):
                    if words[i][0].isupper():
                        company_start_idx = i
                    else:
                        break

                name_words = words[:company_start_idx]
                company_words = words[company_start_idx:]
                # Add back the suffix part
                company_str = ' '.join(company_words) + left[suffix_start - (1 if suffix_start > 0 and left[suffix_start-1] == ' ' else 0):]
                company_str = left[len(' '.join(name_words)):].strip()

                if name_words:
                    participants.append({
                        'name': ' '.join(name_words),
                        'company': company_str,
                        'title': title,
                    })
                    continue

            # Fallback: assume first two words are the name, rest is company
            words = left.split()
            if len(words) >= 3:
                participants.append({
                    'name': ' '.join(words[:2]),
                    'company': ' '.join(words[2:]),
                    'title': title,
                })
            elif len(words) == 2:
                participants.append({
                    'name': left,
                    'company': '',
                    'title': title,
                })
            else:
                participants.append({
                    'name': left,
                    'company': '',
                    'title': title,
                })
            continue

        # No " - " separator at all: treat entire line as a name
        participants.append({
            'name': line,
            'company': '',
            'title': '',
        })
    return participants


def _parse_speaker_turns(body: str) -> List[Dict]:
    """
    Parse a section body (Presentation or Q&A) into sequential speaker turns.

    Returns:
        List of dicts: speaker_line, turn_id, text
    """
    # Find all speaker headers with their positions
    matches = list(SPEAKER_HEADER.finditer(body))
    if not matches:
        return []

    turns = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()

        turns.append({
            'speaker_line': m.group('speaker_line').strip(),
            'turn_id': int(m.group('turn_id')),
            'text': text,
        })
    return turns


def _resolve_speaker(speaker_line: str,
                     corporate_participants: List[Dict],
                     conference_participants: List[Dict]) -> Dict:
    """
    Resolve a speaker line into structured speaker metadata.

    Returns:
        Dict with keys: name, company, title, role
        role is one of: 'operator', 'corporate', 'analyst', 'unknown'
    """
    if OPERATOR_PATTERN.match(speaker_line):
        return {
            'name': 'Operator',
            'company': '',
            'title': 'Operator',
            'role': 'operator',
        }

    m = SPEAKER_DETAIL.match(speaker_line)
    if m:
        name = m.group('name').strip()
        company = m.group('company').strip()
        title = m.group('title').strip()
    else:
        name = speaker_line.strip()
        company = ''
        title = ''

    # Try to match against participant lists
    name_lower = name.lower()

    for p in corporate_participants:
        if p['name'].lower() == name_lower:
            return {
                'name': p['name'],
                'company': p.get('company', company),
                'title': p.get('title', title),
                'role': 'corporate',
            }

    for p in conference_participants:
        if p['name'].lower() == name_lower:
            return {
                'name': p['name'],
                'company': p.get('company', company),
                'title': p.get('title', title),
                'role': 'analyst',
            }

    # Fallback: if speaker has a company-title line, assume corporate
    if company and title:
        return {
            'name': name,
            'company': company,
            'title': title,
            'role': 'corporate',
        }

    return {
        'name': name,
        'company': company,
        'title': title,
        'role': 'unknown',
    }


def _is_ceo(title: str) -> bool:
    """Check if a title string corresponds to a CEO role."""
    title_lower = title.lower().strip()
    for kw in CEO_TITLE_KEYWORDS:
        if kw in title_lower:
            return True
    return False


def parse_earnings_call(text: str) -> Dict:
    """
    Parse a Refinitiv StreetEvents earnings call transcript.

    Args:
        text (str): Full transcript text.

    Returns:
        dict with keys:
            - header (str): Raw header text (event info, date, etc.)
            - corporate_participants (list[dict]): Name/company/title of
              corporate speakers.
            - conference_participants (list[dict]): Name/company/title of
              analysts / conference call participants.
            - presentation (list[dict]): Speaker turns in the Presentation
              section. Each dict: name, company, title, role, turn_id,
              section, text.
            - qa (list[dict]): Speaker turns in the Q&A section.
              Same structure as presentation.
            - all_turns (list[dict]): Combined presentation + qa turns.

    Example:
        >>> transcript = open('call.txt').read()
        >>> result = parse_earnings_call(transcript)
        >>> ceo_qa = [t for t in result['qa']
        ...           if t['is_ceo']]
    """
    sections = _identify_sections(text)

    header = ''
    corporate_participants = []
    conference_participants = []
    presentation_body = ''
    qa_body = ''

    for name, body in sections:
        if name == 'Corporate Participants':
            corporate_participants = _parse_participant_block(body)
        elif name == 'Conference Call Participants':
            conference_participants = _parse_participant_block(body)
        elif name == 'Presentation':
            presentation_body = body
        elif name == 'Questions and Answers':
            qa_body = body
        elif name not in ('Definitions', 'Disclaimer'):
            # Treat unrecognised early sections as header
            if not presentation_body and not qa_body:
                header = (header + '\n' + name + '\n' + body).strip()

    # Parse turns
    def _build_turns(body: str, section_label: str) -> List[Dict]:
        raw_turns = _parse_speaker_turns(body)
        turns = []
        for t in raw_turns:
            speaker = _resolve_speaker(
                t['speaker_line'],
                corporate_participants,
                conference_participants,
            )
            turns.append({
                'name': speaker['name'],
                'company': speaker['company'],
                'title': speaker['title'],
                'role': speaker['role'],
                'is_ceo': _is_ceo(speaker.get('title', '')),
                'turn_id': t['turn_id'],
                'section': section_label,
                'text': t['text'],
            })
        return turns

    presentation = _build_turns(presentation_body, 'Presentation')
    qa = _build_turns(qa_body, 'Q&A')

    return {
        'header': header,
        'corporate_participants': corporate_participants,
        'conference_participants': conference_participants,
        'presentation': presentation,
        'qa': qa,
        'all_turns': presentation + qa,
    }


def get_ceo_qa_text(parsed: Dict) -> str:
    """
    Extract the concatenated text of all CEO turns in the Q&A section.

    This is the primary input for the FPSP narcissism metric, which
    requires unscripted CEO speech only.

    Args:
        parsed (dict): Output of parse_earnings_call().

    Returns:
        str: Concatenated CEO Q&A text, or empty string if no CEO
             turns are found.
    """
    ceo_turns = [t['text'] for t in parsed['qa'] if t['is_ceo']]
    return ' '.join(ceo_turns)


def parse_earnings_call_to_df(text: str) -> pd.DataFrame:
    """
    Parse a Refinitiv transcript and return all turns as a DataFrame.

    Columns: name, company, title, role, is_ceo, turn_id, section, text

    Args:
        text (str): Full transcript text.

    Returns:
        pd.DataFrame
    """
    parsed = parse_earnings_call(text)
    if not parsed['all_turns']:
        return pd.DataFrame(
            columns=['name', 'company', 'title', 'role',
                     'is_ceo', 'turn_id', 'section', 'text']
        )
    return pd.DataFrame(parsed['all_turns'])


def extract_header_metadata(text: str) -> Dict[str, str]:
    """
    Extract event metadata from the transcript header.

    Attempts to parse:
        - quarter (e.g. "Q3 2023")
        - company_name (e.g. "Agilent Technologies Inc")
        - event_type (e.g. "Earnings Call")
        - date (e.g. "AUGUST 15, 2023")

    Args:
        text (str): Full transcript text.

    Returns:
        dict with keys: quarter, company_name, event_type, date (all str,
        may be empty if not found).
    """
    metadata = {
        'quarter': '',
        'company_name': '',
        'event_type': '',
        'date': '',
    }

    # Look for the event line, e.g.:
    # "Q3 2023 Agilent Technologies Inc Earnings Call"
    event_pattern = re.compile(
        r'(Q[1-4]\s+\d{4})\s+'       # Quarter
        r'(.+?)\s+'                    # Company name
        r'(Earnings\s+Call|'           # Event type variants
        r'Conference\s+Call|'
        r'Earnings\s+Conference)',
        re.IGNORECASE
    )

    # Search first 20 lines of the text
    header_lines = text.split('\n')[:20]
    header_text = '\n'.join(header_lines)

    m = event_pattern.search(header_text)
    if m:
        metadata['quarter'] = m.group(1).strip()
        metadata['company_name'] = m.group(2).strip()
        metadata['event_type'] = m.group(3).strip()

    # Look for a date line, e.g. "AUGUST 15, 2023 / 8:30PM GMT"
    date_pattern = re.compile(
        r'([A-Z]+\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE
    )
    dm = date_pattern.search(header_text)
    if dm:
        metadata['date'] = dm.group(1).strip()

    return metadata
