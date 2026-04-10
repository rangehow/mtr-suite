from langchain_text_splitters import RecursiveCharacterTextSplitter

def group_texts(examples, max_length=4096):
 result = {key: [] for key in examples}
 text_splitter = RecursiveCharacterTextSplitter(
 separators=[
 "\n\n",
 "\n",
 ".",
 "!",
 "?",
 ",",
 " ",
 "\u200b", # Zero-width space
 "\uff0c", # Fullwidth comma
 "\u3001", # Ideographic comma
 "\uff0e", # Fullwidth full stop
 "\u3002", # Ideographic full stop
 "",
],
 chunk_size=max_length,
 chunk_overlap=0,
 length_function=len,
 is_separator_regex=False,
)

 texts = examples['text']
 for i in range(len(texts)):
 text = texts[i]
 sub_texts = text_splitter.split_text(text)

 for sub_text in sub_texts:
 # \n\n 
 if sub_text.endswith('\n\n'):
 # \n\n, 
 sub_text = sub_text[:-2]
 else:
 # \n\n 
 last_double_newline = sub_text.rfind('\n\n')
 if last_double_newline!= -1:
 # \n\n 
 tail = sub_text[last_double_newline+2:]
 if len(tail.split()) <= 5:
 sub_text = sub_text[:last_double_newline]
 
 if len(sub_text) >= max_length//2:
 for key in examples:
 if key == 'text':
 result['text'].append(sub_text)
 else:
 result[key].append(examples[key][i])

 return result


if __name__=='__main__':

 text='''Sylvan Lake is located next to the community by that name in the Town of Beekman, New York, United States. It is the deepest and second-largest lake in Dutchess County.\n\nIt is a popular local recreation spot. Many summer camps are located along it, as well as a large private campground. Many residents from Chelsea Cove also use the lake recreationally.\n\nGeography\n\nIt is an irregular rounded shape, with a surface area, located near the town's southwestern border with East Fishkill. High unnamed hills rise to to its northeast, above the lake. The unincorporated hamlet of Sylvan Lake is located to its southeast and east. Dutchess County Route 10 runs northwest from the hamlet along the lakeshore, providing access to the nearby Taconic State Parkway via the NY 82 state highway at Arthursburg.\n\nTo the southwest are lower hills. The lake's unnamed outlet brook flows at first northward through a wetland between two smaller hills at that end but soon turns southward, carrying the drainage from the lake and its basin south a few miles to Fishkill Creek. From there it reaches the Hudson River at Beacon.\n\nThe lake is located adjacent to the Chelsea Cove housing complex and all its residents are permitted access to the lake. There is a small beach with a small area to wade in before it drops off into much deeper waters. Sylvan Lake Beach Park, a privately operated campground, is located along another shore. The remainder of the shoreline is owned by private residences and summer camps.\n\nHydrology\n\nThe lake is deep, the deepest in Dutchess County. Formerly a quarry used for mining, the slope is very steep and reaches maximum depth quickly near the center of the lake. The thermocline in the lake reaches about 40 degrees F. To scuba dive in the lake, advanced open water certification is needed. Equipment such as 7mml full body wetsuits including gloves and hood are necessary as are lights due to the murkiness of the water.\n\nRecreation'''

 text_splitter = RecursiveCharacterTextSplitter(
 separators=[
 "\n\n",
 "\n",
 ".",
 "!",
 "?",
 ",",
 " ",
 "\u200b", # Zero-width space
 "\uff0c", # Fullwidth comma
 "\u3001", # Ideographic comma
 "\uff0e", # Fullwidth full stop
 "\u3002", # Ideographic full stop
 "",
],
 chunk_size=1200,
 chunk_overlap=0,
 length_function=len,
 is_separator_regex=False,
)

 print(text_splitter.split_text(text))