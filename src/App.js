import React,{ Component } from 'react';
import { withRouter } from "react-router";
import { v4 as uuid } from 'uuid';
//import TextareaAutosize from 'react-textarea-autosize';
import './App.css';

import Speech from "./Speech";
import TextInput from "./TextInput";
const dialogflow = require('dialogflow').v2;
const auth = require('google-auth-library');
const secret = require('./secret.json');

const client_id = secret.client_id;
const client_secret = secret.client_secret;
// Local machine call back url
//const callback_uri = "http://localhost:3000/"
// AWS Amplify master call back url
const callback_uri = "https://master.d1c4rg4lvti4vl.amplifyapp.com/"
const oauth2client = new auth.OAuth2Client(client_id, client_secret, callback_uri);
const projectId = 'gossipsiri';
const welcomeText = 'Wall Street Digest Trading Assistant Here. How can I help?';
const s3_bucket_link = "https://wall-street-digest-media.s3.us-east-2.amazonaws.com/"

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { chatHistory: [], inputText: '', sessionsClient: {}, speechMode: false, imgUrl:''};
    // Initialize a new session if current session is not yet set
    if (sessionStorage.getItem('sessionId') === null) {
      sessionStorage.setItem('sessionId', uuid());
    }
    if (sessionStorage.getItem('chatHistory') === null) {
      sessionStorage.setItem('chatHistory',
        JSON.stringify([{type: 'output', createdTime: Date.now(), value: welcomeText}]));
    }
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleSwitch = this.handleSwitch.bind(this);
    this.handleInput = this.handleInput.bind(this);
  }

  componentDidMount() {
    this.handleAuth();
  }

  async handleSubmit() {
    const {inputText, sessionsClient} = this.state;
    const chatWindow = document.getElementById("chatWindow");
    const trimmed_message = inputText.trim();
    if (trimmed_message === '') {
      window.alert("Unable to send blank message");
      this.setState({inputText: ''});
    } else if (trimmed_message.length > 256) {
      window.alert("Message too long (> 256 characters)");
      this.setState({inputText: ''});
    } else {
      const sessionPath = sessionsClient.sessionPath(projectId, sessionStorage.getItem('sessionId'));
      // The text query request.
      const request = {
        session: sessionPath,
        queryInput: {
          text: {
            // The query to send to the dialogflow agent
            text: trimmed_message,
            // The language used by the client (en-US)
            languageCode: 'en-US',
          },
        },
      };
      console.log(request)
      const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory'));
      sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory.concat(
        {type: 'input', createdTime: Date.now(), value: trimmed_message}
      )));
      // Re-render and Auto scroll down after sending new message
      this.setState({ state: this.state }, () => {
        chatWindow.scrollTop = chatWindow.scrollHeight;
      });
      sessionsClient.detectIntent(request)
        .then((responses) => {
          console.log('Detected intent');
          const result = responses[0].queryResult;
          console.log(result.fulfillmentMessages)
          console.log(`  Query: ${result.queryText}`);
          console.log(`  Response: ${result.fulfillmentText}`);
          if (result.intent) {
            console.log(`  Intent: ${result.intent.displayName}`);
            if (result.intent.displayName.includes('plot') &&
                !result.fulfillmentText.startsWith('Please specify')) {
              console.log("Plot here")
              // Add response to chatHistory
              const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory'));
              sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory.concat(
                  {type: 'output_img', createdTime: Date.now(), value: result.fulfillmentText}
              )));
              this.setState({
                chatHistory: this.state.chatHistory
                    .concat({type: 'output_img', createdTime: Date.now(), value: result.fulfillmentText})
              }, () => {
                // Auto scroll down after receiving new message
                chatWindow.scrollTop = chatWindow.scrollHeight;
              })
            } else {
              // Add response to chatHistory
              const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory'));
              sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory.concat(
                  {type: 'output', createdTime: Date.now(), value: result.fulfillmentText}
              )));
              this.setState({
                chatHistory: this.state.chatHistory
                    .concat({type: 'output', createdTime: Date.now(), value: result.fulfillmentText})
              }, () => {
                // Auto scroll down after receiving new message
                chatWindow.scrollTop = chatWindow.scrollHeight;
              })
            }
          } else {
            console.log(`  No intent matched.`);
          }
        })
        //Redirect to Auth on request rejection
        .catch(e => {
          console.log(e);
          if (e.code === 16) {
            this.handleAuth();
          }
        });
    }
  }

  redirectAuth() {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    // redirect user to authUrl and wait for them coming back to callback_uri
    const authUrl = oauth2client.generateAuthUrl({
      access_type: 'offline',
      scope: [    // scopes for Dialogflow
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/dialogflow'
      ]
    });
    window.location.href = authUrl;
  }

  async handleAuth() {
    // Get code from URL query
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const code = urlParams.get('code');
    try {
      if (sessionStorage.getItem('accessToken') === null) {
        if (code === null) {
          this.redirectAuth();
        } else {
          const tokenResponse = await oauth2client.getToken(code);
          sessionStorage.setItem('accessToken', tokenResponse.tokens.access_token);
          sessionStorage.setItem('refreshToken', tokenResponse.tokens.refresh_token);
          oauth2client.setCredentials({
            access_token: sessionStorage.getItem('accessToken')
          });
        }
      } else {
        const currentTokenInfo = await oauth2client.getTokenInfo(sessionStorage.getItem('accessToken'));
        // if token expired, try to use refresh_token
        if (currentTokenInfo.expiry_date < Date.now()) {
          if (sessionStorage.getItem('refreshToken') !== null) {
            oauth2client.setCredentials({
              refresh_token: sessionStorage.getItem('refreshToken')
            });
          } else {
            // if no refresh_token existed, redo authentication
            this.redirectAuth();
          }
        } else {
          oauth2client.setCredentials({
            access_token: sessionStorage.getItem('accessToken')
          });
        }
      }
      console.log(oauth2client);
      //https://github.com/googleapis/nodejs-dialogflow/issues/405#issuecomment-529713296
      global.isBrowser = true;
      const sessionsClient = new dialogflow.v2.SessionsClient({auth: oauth2client});
      this.setState({
        sessionsClient: sessionsClient
      })
    } catch (e) {
      console.log(e);
      // If any error in auth process, redirect to auth
      this.redirectAuth();
    }
  }

  // Start a new session
  handleNewSession() {
    // Clean chatHistory and start a new session
    sessionStorage.setItem('sessionId', uuid());
    sessionStorage.setItem('chatHistory',
      JSON.stringify([{type: 'output', createdTime: Date.now(), value: welcomeText}]));
    // Force re-render
    this.setState({ state: this.state });
  }

  // Switch between text/voice input mode
  handleSwitch() {
    this.setState(prevState => {
      return {
        speechMode: !prevState.speechMode
      }}, () => {
      if (this.state.speechMode) {
        document.getElementById("switch").innerHTML = "Text Mode";
      } else {
        document.getElementById("switch").innerHTML = "Voice Mode";
      }
    });
  }

  handleInput(inputText) {
    this.setState({inputText: inputText}, this.handleSubmit);
  }

  render() {
    const {speechMode} = this.state;
    const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory'));
    let inputComponent;
    if (speechMode) {
      inputComponent = <Speech onSpeech={this.handleInput} />
    } else {
      inputComponent = <TextInput onTextInput={this.handleInput}/>
    }
    return (
      <div className="App">
        <header>
          <img className="inlineLogo" src={"wsd_logo.png"} alt="Logo" />
        </header>
        <div style={{width: '100%', height:'100%', position: 'absolute'}}>
          <div id='chatWindow' style={{width: '84%', height: 500, marginLeft:'8%',
            marginTop:'1%', marginBottom:'2%', overflow: 'scroll'}}>
            {chatHistory.map((message) => {
              if (message.type === 'input') {
                return <div className='bubbleWrapper'>
                  <div className="inlineContainer own">
                    <img className="inlineIcon" src={"user.png"} alt="User Icon"/>
                    <div className='userBubble' key={message.createdTime}>
                      {message.value}
                    </div>
                  </div>
                </div>
              } else if (message.type === 'output_img') {
                return <div className='bubbleWrapper'>
                  <div className="inlineContainer">
                    <img className="inlineIcon" src={"bot.png"} alt="Bot Icon" />
                    <div className='botBubble' key={message.createdTime}>
                      <img src={s3_bucket_link + message.value} alt=""/>
                    </div>
                  </div>
                </div>
              } else {
                return <div className='bubbleWrapper'>
                  <div className="inlineContainer">
                    <img className="inlineIcon" src={"bot.png"} alt="Bot Icon" />
                    <div className='botBubble' key={message.createdTime}>
                      {message.value}
                    </div>
                  </div>
                </div>
              }
            })}
          </div>
          <button id='switch' className='button' onClick={this.handleSwitch}>Speech Input</button>
          <button id='clear' className="button" onClick={() => {
            this.handleNewSession()
          }}>Clear
          </button>
          <div style={{position:'relative', left:'30%', top:'2%', width:'40%'}}>
            {inputComponent}
          </div>
          <br/><br/><br/><br/><hr/>
          <p id='ack1'><small>The content is for informational and educational purposes only and should not be construed as investment
            advice or an offer or solicitation in respect to any products or services for any persons who are prohibited
            from receiving such information under the laws applicable to their place of citizenship, domicile or residence.
          </small></p>
          <p id='ack2'><small>The content presented does not constitute investment advice, should not be used as the basis for any
            investment decision, and does not purport to provide any legal, tax or accounting advice. Please remember
            that there are inherent risks involved with investing in the markets, and your investments may be worth more
            or less than your initial investment upon redemption. There is no guarantee that Wall Street Digest's
            objectives will be achieved.
          </small></p>
        </div>
      </div>
    );
  }
}

export default withRouter(App);
