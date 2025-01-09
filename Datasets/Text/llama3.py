import pandas as pd
import re
import signal
import time  
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm  

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
model_name = 'llama3'


def truncate_content(content, max_words=3500):
    words = content.split()
    if len(words) > max_words:
        truncated_content = ' '.join(words[:max_words])
        return truncated_content
    return content


def label_and_reason(content):
    truncated_content = truncate_content(content)
    full_prompt = f"""Assess the text below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.
                If you find any of the listed rhetorical techniques, then the article is likely disinformation; if not, it is likely not disinformation.
                Provide three separate assessments with 'Likely' or 'Unlikely' followed by one-line long concise reasoning on why you chose 'Likely' or 'Unlikely' for each without any further explanation.
                
                Rhetorical Techniques Checklist:
                - Emotional Appeal: Uses language that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.
                - Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.
                - Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.
                - Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.
                - Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.
                - Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.
                - Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.
                - Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.
                - Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.
                - Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.
                - Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.
                - Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.
                
                Response format required: 
                1. [Likely/Unlikely] [Reasoning], 2. [Likely/Unlikely] [Reasoning], 3. [Likely/Unlikely] [Reasoning]

                Text: {truncated_content}"""
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, temperature=0.2, max_new_tokens=500)  
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def convert_analysis_to_dataframe(row, analysis_str):
    results = row.to_dict()
    results[f'{model_name}'] = ''

    analysis_str = analysis_str.replace('\n', ' ').replace('\r', ' ').strip()

    normalized_str = analysis_str.lower()

    pattern = r'[\[\(\{\<]\s*\b(likely|unlikely)\b\s*[\]\)\}\>]|\b(likely|unlikely)\b'
    matches = re.findall(pattern, normalized_str)

    label_count = {'Likely': 0, 'Unlikely': 0}
    
    for match in matches:
        label = match[0].strip().capitalize() if match[0] else match[1].strip().capitalize()
        if label in label_count:
            label_count[label] += 1

    if label_count['Likely'] > label_count['Unlikely']:
        results[f'{model_name}'] = 'Likely'
    elif label_count['Unlikely'] > label_count['Likely']:
        results[f'{model_name}'] = 'Unlikely'
    else:
        print(f"No label Assigned for {results['unique_id']}")
        return None

    return results


def handle_signal(signal_num, frame):
    print(f"\nSignal {signal_num} received. Saving progress before exiting.")
    raise KeyboardInterrupt if signal_num in {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1} else SystemExit


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP, signal.SIGUSR1):
    signal.signal(sig, handle_signal)


def process_csv(batch_size=100):

    data = pd.read_csv("/fs01/projects/NMB-Plus/Caesar/Datasets/text_merged.csv")

    processed_file = f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"
    if os.path.exists(processed_file):
        processed_data = pd.read_csv(processed_file, usecols=['unique_id'])  
        processed_indices = set(processed_data['unique_id'].values)  
    else:
        processed_indices = set()
    print("Processed Unique ID Numbers: ", len(processed_indices))

    data = data[~data['unique_id'].isin(processed_indices)]

    results_list = []

    try:
        for index, row in tqdm(data.iterrows(), total=len(data), desc='Processing'):
            unique_id = row.get('unique_id', None)
            
            content = row.get('content', '').replace('\n', ' ').replace('\r', ' ').strip()

            if content:
                start_time = time.time()  
                analysis_result = label_and_reason(content)
                
                if analysis_result:
                    result = convert_analysis_to_dataframe(row, analysis_result)
                    if result:
                        results_list.append(result)

                elapsed_time = time.time() - start_time
                print(f"Processed unique_id: {unique_id}, Time taken: {elapsed_time:.2f} seconds.")


            if len(results_list) >= batch_size:
                pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"), index=False)
                results_list.clear()

    except (KeyboardInterrupt, SystemExit):  
        print("\nProcess interrupted or suspended! Saving current progress...")
    
    finally:
        if results_list:
            pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"), index=False)
            print(f"Final results saved to /fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv.")


def test_single(content):
    analysis_result = label_and_reason(content)
    print("analysis result:", analysis_result)
    if analysis_result:
        dummy_row = pd.Series({'unique_id': 'test_id', 'content': content})
        result = convert_analysis_to_dataframe(dummy_row, analysis_result)
        print("Single analysis result:", result)
    else:
        print("Analysis failed for the provided content.")


if __name__ == "__main__":
    test_content = "The audience inside the Chicago Bulls’ home arena erupted in cheers when Vice President Kamala Harris walked on stage to accept her party’s nomination for President. Thousands of delegates wore white in honor of the American women who fought for the right to vote a century ago. The sea of suffragette fabric in front of Harris served as a visual reminder of historic possibility: a Black woman of South Asian descent with an odds-on chance to be the American President.  Harris started off by acknowledging the improbability of the moment. Just a month before, President Joe Biden dropped out and endorsed her run, and in four weeks she has retooled his campaign into her own. “But I’m no stranger to unlikely journeys,” she said. Her path through life, she said, going from being the daughter of two immigrant students who met and fell in love at a civil rights gathering to being elected to the U.S. Senate and then becoming Vice President, is a story that “could only be written in the greatest nation on Earth.”  In a 40-minute speech, Harris described the lessons she learned from her parents and made a forceful case against Donald Trump, promising “a chance to chart a new way forward.” She called Trump an “unserious man” who brought “chaos and calamity” to the White House once before, and when he lost the 2020 election “tried to throw away your votes” by encouraging a mob to storm the U.S. Capitol. As President, Harris said she would work to bring the country together to create jobs, lower costs, and create an “opportunity economy.”  Here are the highlights of her historic acceptance speech:  ‘A President for all Americans’  As she accepted the Democratic nomination, Harris made a direct appeal to Republicans disillusioned by former President Donald Trump and undecided voters, vowing to be a President for all Americans. “I know there are people of various political views watching tonight,” she said.  “And I want you to know: I promise to be a President for all Americans,” Harris added, drawing a contrast to Trump. “You can always trust me to put country above party and self.”  “My entire career, I’ve only had one client: The People,” she said, a nod to her background as a prosecutor. “And so, on behalf of The People; on behalf of every American, regardless of party, race, gender, or the language your grandmother speaks; on behalf of my mother and everyone who has ever set out on their own unlikely journey; on behalf of Americans like the people I grew up with, people who work hard, chase their dreams, and look out for one another; on behalf of everyone whose story could only be written in the greatest nation on Earth—I accept your nomination to be President of the United States of America.”  ‘Never do anything half-assed’  Harris’ mother Shyamala Gopalan died in 2009, but many of the lessons she taught Harris have stuck with her. Harris said her mother raised her and her sister Maya to “never complain about injustice” but to “do something about it.”  Gopalan came to the U.S. from India at the age of 19 to be a scientist working on a cure for breast cancer and fell in love with a Jamaican student named Donald Harris when the two met at a civil rights gathering. The two had Kamala and her sister Maya and moved frequently before the marriage broke up and her mother raised her two daughters largely on her own within a tight community of friends in a working-class Oakland neighborhood.  Harris remembered her mother working long hours and close friends from the neighborhood looking after them, teaching them how to make gumbo and play chess. Her house was full of the music of Aretha Franklin, John Coltrane, and Miles Davis, and conversations about civil rights leaders and lawyers like Thurgood Marshall and Constance Baker Motley, she said.  Her mother was a “brilliant, five-foot-tall, brown woman with an accent. And, as the eldest child, I saw how the world would sometimes treat her,” Harris said. But her mom “never lost her cool” and was “tough” and “courageous” and “a trailblazer for womens’ health.”  There was another lesson her mother taught her, Harris said: “Never do anything half-assed.”  ‘Run, Kamala, run. Don’t be afraid’  Harris’ father, Donald Harris, eventually became a professor of economics at Stanford University, but after her parents split when Harris and her sister were in grade school, it was mostly her mother who raised them.  Harris recalled her father teaching her from her earliest years “to be fearless.” On trips to the park, her mother would often tell Harris to stay close, she said. But her father had a different take. He would smile, she said, and tell her, “Run, Kamala, run. Don’t be afraid. Don’t let anything stop you.”  ‘One of the reasons I became a prosecutor’  Harris said she was inspired to become a prosecutor to protect victims after helping a friend in high school who was being sexually abused by her step father.  “When I was in high school, I started to notice something about my best friend Wanda. She was sad at school, and there were times she did not want to go home. So, one day, I asked if everything was alright. She confided in me that she was being sexually abused by her step-father,” Harris said. “I immediately told her she had to come stay with us. And she did.”  “This is one of the reasons I became a prosecutor, to protect people like Wanda, because I believe everyone has a right to safety, to dignity, and to justice,” she added.  Harris went to law school at the University of California Hastings and then served three decades as a prosecutor, working as a deputy district attorney in Alameda County and eventually San Francisco’s district attorney. In 2011, she was elected attorney general of California.  “As a prosecutor, when I had a case, I charged it not in the name of the victim. But in the name of ‘The People.’ For a simple reason: In our system of justice, a harm against any one of us is a harm against all of us. And I would often explain this to console survivors of crime, to remind them: No one should be made to fight alone. We are all in this together,” she said.  “And every day in the courtroom, I stood proudly before a judge, and I said five words: ‘Kamala Harris, for the People.’”  ‘Working to end [the Israel-Hamas] war’  “Let me be clear,” Harris said, addressing one of the most controversial topics looming over the convention, “I will always stand up for Israel’s right to defend itself, and I will always ensure Israel has the ability to defend itself, because the people of Israel must never again face the war that a terrorist organization called Hamas caused on October 7, including unspeakable sexual violence and the massacre of young people at a music festival.”  “At the same time, what has happened in Gaza over the past 10 months is devastating. So many innocent lives lost. Desperate, hungry people fleeing for safety, over and over again. The scale of suffering is heartbreaking.”  “President Biden and I are working to end this war such that Israel is secure, the hostages are released, the suffering in Gaza ends, and the Palestinian people can realize their right to dignity, security, freedom, and self-determination.”  ‘Trump is an unserious man’  Harris explained the consequences of electing Trump, pointing to Project 2025 as an example of what he would do in a second term. “In many ways, Donald Trump is an unserious man. But the consequences of putting Donald Trump back in the White House are extremely serious,” she said, adding that the election is a “fight for America’s future.”  “Consider not only the chaos and calamity of when he was in office but also the gravity of what has happened since he lost the last election,” Harris said.  She spoke about Trump’s role on Jan. 6, sending an armed mob to the Capitol where they assaulted law enforcement officers, saying that she would respect “the rule of law to free and fair elections” and “the peaceful transfer of power.” She also highlighted Trump’s conviction on 34 felony counts and being found liable for sexual abuse, and mentioned the Supreme Court’s recent ruling that former Presidents can claim immunity from prosecution for official actions taken in office—a case argued by Trump’s lawyers. “Just imagine Donald Trump with no guardrails,” she said.  “And how he would use the immense powers of the United States. Not to improve your life. Not to strengthen our national security. But to serve the only client he has ever had: himself.”  ‘Will create an opportunity economy’  Harris promised to build an “opportunity economy” that allows more people “a chance to compete and a chance to succeed whether you live in a rural area, small town, or big city.”  In Harris’s childhood home, her mother kept a strict budget and they lived within their means, she said. But her mom also expected her kids to make the most of the opportunities because “opportunity is not available to everyone.”  Harris said she would work to bring workers, small business owners, entrepreneurs, and American companies together to create jobs and grow the economy. She said she would work to lower the costs of healthcare, housing, and groceries. As part of her economic plan, Harris has previously promised to block companies from taking advantage of crises to raise prices. She also promised to “end” the country’s housing shortage and protect Social Security and Medicare payments from cuts.  Anti-abortion Republicans ‘are out of their minds’  Harris outlined several ways that Trump and his allies would restrict women’s reproductive rights. She claimed that they would limit access to birth control, ban medication abortion, enact a nationwide abortion ban, force states to report on women’s miscarriages and abortions “with or without Congress,” and create a national anti-abortion coordinator.  ​​“Simply put,” she said, “they are out of their minds.”  Trump has previously said he would not sign a national abortion ban into law if he were in the White House again, saying the decision to restrict abortion should be left to the states.  “Why exactly is it that they don’t trust women?” Harris asked. “Well, we trust women,” she said, vowing to sign into law a bill to restore reproductive freedom if Congress passes one.  ‘Refuse to play politics with our security’  Harris said that if she wins the election she would bring back the bipartisan border security bill that Trump and his allies tanked, and sign it into law.  “We can create an earned pathway to citizenship and secure our border,” Harris said, adding that Republicans and Democrats wrote the border bill together and it was endorsed by the Border Patrol. Trump lobbied Republicans to vote against the bill, in part because he didn’t want to deliver Biden a political victory ahead of the election on an issue to which Democrats are politically vulnerable.  “I refuse to play politics with our security. And here is my pledge to you: As President, I will bring back the bipartisan border security bill that he killed and I will sign it into law,” Harris said. “I know we can live up to our proud heritage as a nation of immigrants and reform our broken immigration system. We can create an earned pathway to citizenship and secure our border.”  ‘Trump won’t hold autocrats accountable, because he wants to be an autocrat himself’  Harris described meeting in person with Ukrainian President Volodymyr Zelensky five days before Russia invaded in 2022 and warning him about Putin’s plans. She promised Thursday that as President she would continue to stand strong with Ukraine and the North Atlantic Treaty Organization alliance and “ensure America always has the strongest, most lethal fighting force in the world.”  She attacked Trump’s threats to abandon NATO allies and his penchant for complimenting dictators like Russian President Vladimir Putin and North Korea’s Supreme Leader Kim Jong Un. Trump “encouraged Putin to invade our allies. Said Russia could ‘do whatever the hell they want,’” Harris said.  Those dictators are “rooting for Trump, because they know he is easy to manipulate with flattery and favors. They know Trump won’t hold autocrats accountable, because he wants to be an autocrat himself,” she said, to loud cheers in the arena.  ‘Never let anyone tell you who you are’  As other speakers had done throughout the DNC, Harris called out Trump and his running mate J.D. Vance for “denigrating America” in their messaging on the campaign trail.  “Our opponents in this race are out there every day denigrating America, talking about how terrible everything is,” she said. “Well, my mother had another lesson she used to teach: Never let anyone tell you who you are; you show them who you are.”  Harris told voters the election is a chance to act on her mother’s advice. “America,” she said, “let us show each other and the world who we are and what we stand for: freedom, opportunity, compassion, dignity, fairness, and endless possibilities.”"
    test_single(test_content)
    test_content2 = "Chicago – Not too long ago, Kamala Harris was facing one of the biggest tests of her political career. Her boss had completely fumbled on the debate stage for the entire world to see, and it was her job to defend his performance – and presidency — to the American people.  “Yes,” Joe Biden had a “slow start” to the debate that was “obvious to everyone,” the media-shy vice president said in a post-debate interview with CNN on July 27. But, she added, “I’m not going to spend all night with you talking about the last 90 minutes when I’ve been watching the last three and a half years of performance.”  Out with the old and in with his second-in-command.  “America, the path that led me here in recent weeks was no doubt unexpected. But I’m no stranger to unlikely journeys,” Harris told the crowd. “My mother Shyamala Harris had one of her own. I miss her every day–especially now. And I know she’s looking down tonight, and smiling. My mother was 19 when she crossed the world alone, traveling from India to California with an unshakeable dream to be the scientist who would cure breast cancer.”  These days, Kamala Harris really does face the biggest test of her political life as she and her new running mate, Minnesota Governor Tim Walz, take the mantle for the Democratic Party in its fight against Donald Trump. Her remarks were upbeat and patriotic, leaning into her background as a prosecutor taking on sexual abusers and human traffickers.  She told the crowd that in her entire career, she has only ever had “one client — the people.”  Harris, who has rarely waded into foreign policy from the stump as vice president, made a point to address the importance of preserving American national security by projecting strength abroad. She pledged to uphold the country’s “sacred” obligation to honor and care for veterans, help the U.S. compete with China, “stand strong with Ukraine and our NATO allies,” and continue fighting for a cease-fire in Gaza.  And she cast her opponent as a threat to democracy and national security. “In many ways,” she told the crowd, Trump is an “unserious man.” But the consequences of putting him back in the White House, are “extremely serious.” She assailed his behavior leading up to and following the storming of the U.S. Capitol three years ago. “When politicians in his own party begged him to call off the mob and send help, he did the opposite – he fanned the flames.”  She spoke of his convicted by a Manhattan jury of “everyday Americans” and his conviction a separate sexual-assault trial.  “Just imagine Donald Trump with no guardrails, and how he would use the immense powers of the presidency of the United states. Not to improve your life, not to strengthen national security,” she said. “But to serve the only client he has ever had – himself.”  Things weren’t always looking up for Harris. Winning this November would mark quite the turnaround for this vice president, whose approval numbers hovered even below Biden’s for much of his presidency until she was crowned his replacement.  Soon after she was elected vice president, story after story unveiled the dysfunction that plagued her office and the frustrations that surrounded her team over her lack of policy portfolio under the Biden administration. That’s all water under the bridge for Democrats these days.  Throughout the week, delegates and politicians sang her praises as a former prosecutor, attorney general, and U.S. Senator with the skill and experience to take on Trump. Harris is bringing back “joy,” will protect “freedom,” and will keep the country from “going back” to Trump.  But if she wins, how will she govern? Last Friday, Harris announced an economic platform to lower the cost of prescription drugs for all Americans, end corporate “price gouging,” and expand affordable housing.  On the stump, she tells voters she will codify Roe v. Wade into law, protect voting rights, and keep Americans safe from gun violence. She touched on all of those themes Thursday evening, promising to cut taxes for the middle class and to create a “opportunity economy” where everyone has the chance to “compete” and “succeed.”  As of today, the Harris-Walz campaign website still contains no issues or policy section. And according to the New York Times, “campaign aides say Ms. Harris intends to release a few targeted policy proposals” but is “unlikely to detail a broader agenda beyond what Mr. Biden has already articulated.”  Here in Chicago this week, Democrats are confident in their new nominee. Harris has been part of an administration “that has saved our economy from destruction in the aftermath of COVID,” Representative Jim McGovern (D., Mass.), the ranking member of the House Rules Committee, told National Review Thursday evening inside the convention arena. She has spent her entire career “fighting for economic justice.”  No nominee can do a thousand things all once, he said, especially one who emerged as a candidate just one month ago. “So people should be a little patient. But if you want to know where she stands on some of these issues, look at her record. It’s only been a few weeks, and so she has to play catch up in terms of getting all the position papers up.”  And what does he make of criticisms from the right that Harris is a shape-shifter on policy? “The right wing should be a little humble right now, a little quiet, given who their candidate is. You want to talk about flip flops and constantly changing his political stripes over his lifetime?”  Donald Trump fits the bill, he said. “Their candidate is imploding and it’s sinking like a like a rock.”  Most people have two years to develop campaign websites and policy positions, former Senator Doug Jones (D., Ala.) said in an interview with National Review inside the United Center Thursday evening. “She assumed this mantle not too long ago, and said she wants to make sure she gets it right. She wants to make sure that what she says she can absolutely defend,” he said.”  The goal this week, Jones continued, is to celebrate her nomination and to tell her story to the American people.  “She’s going to continue to carry the message of hope, of joy, of going forward, not backward, because at the end of the day, policies mean a lot, and they’re what really form the basis,” he said. “But at the same time, people want to feel good about a candidate. They want to feel good about a ticket. They want to know that they can trust them.”  She hopes to earn that trust by casting herself as a bipartisan candidate who will put party above partisanship — no small feat for a nominee who carries the baggage of an unpopular administration. For now, she’s hoping she can ride the high of this week’s raucous energy well into the fall.  “I know there are people of various political views watching tonight. And I want you to know: I promise to be a President for all Americans,” Harris told the crowd Thursday night. “I will be a President who unites us around our highest aspirations. A President who leads — and listens. Who is realistic. Practical. And has common sense. And always fights for the American people. From the courthouse to the White House, that has been my life’s work.”"
    test_single(test_content2)
    # process_csv()