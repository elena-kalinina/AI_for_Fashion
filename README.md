# AI_for_Fashion
this is the code for [Google Hackathon](https://developers.google.com/womentechmakers/initiatives/she-builds-ai?hl=it) submission. 

# Covered use cases
The code covers several use cases related to the analysis of photos from Instagram fashion influencers.  
Several influencers were selected for the photos: 
 - [@jeanne_andreaa](https://www.instagram.com/jeanne_andreaa/)
 - [@pauluschkaa](https://www.instagram.com/pauluschkaa/)
 - [@stephaniebroek](https://www.instagram.com/stephaniebroek/)
 - [@lara_bsmnn](https://www.instagram.com/lara_bsmnn/).

   Below each of the covered use cases is described in detail.
   
## Use case 1: Get an outfit suggestion based on your request

- you can download a bunch of images from instagram and create a dataset where you can search for outfit ideas based on natural langage requests, such as outfit in a certain style (e.g. urban chic), outfit containing certain elements (e.g. ankle boots) or colors (e.g. olive green).

For this purpose, a selection of images was described by Gemini; it identified the outfit style and its key elements. These descriptions are then embedded and out in a dataframe that contains image identifiers along with textual descriptions. When a user asks for a suggestion, her request is qlso embedded, and an image is retrieved whose style and description are closest to the request. To see how the dataset is created, use 0 as use case (see [Running the code](#markdown-header-running-the-code)).

## Use case 2: An outfit in the style of your favourite influencer

- you can download some photos of an instagram user and ask Gemini to suggest an outfit recreating her style. Then, you can compose this outfir out of the items found on Vinted.
  
## Use case 3: Current trends from photos

- you can download a bunch of images from Instagram and ask Gemini to provide a summary of the current trends;

## Use case 4: Recreate an outfit with items from Vinted

- you can upload a photo and recreate the outfit on Vinted (the photo will be described by Gemini in such a way that it will be possible to search for single elements on Vinted).

  The paths in the code are set for each example of the use cases.

  # Running the code

  - Install the requirements:
 
    ```
    pip install -r requirements.txt
    ```

    - Run the file ```ai_for_fashion.py``` indicating the number of the use case as ```-c``` parameter:
   
      ```
      python ai_for_fashion.py -c <number between 0 and 4>
      ```
       



