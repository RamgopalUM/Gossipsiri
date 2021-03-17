import React, { Component } from 'react'

//-----------------SPEECH RECOGNITION SETUP---------------------

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
console.log(SpeechRecognition);
const recognition = new SpeechRecognition();

recognition.continous = true
recognition.lang = 'en-US'

//------------------------COMPONENT-----------------------------

class Speech extends Component {

  constructor(props) {
    super(props);
    this.state = {
      listening: false,
      finalTranscript: '',
      buttonValue: 'Talk'
    }
    recognition.onend = () => {
      console.log("Ended!")
    }
    recognition.stop();
    this.handleClick = this.handleClick.bind(this);
    this.handleCancel = this.handleCancel.bind(this);
    this.handleListen = this.handleListen.bind(this);
  }

  handleClick() {
    this.setState({
      listening: !this.state.listening
    }, this.handleListen);
  }

  handleListen() {
    // handle speech recognition here
    if (this.state.listening) {
      this.setState({buttonValue: "Send"});
      recognition.start();
      recognition.onend = () => {
        console.log("Ended!");
        recognition.start();
      }
    } else {
      this.setState({buttonValue: "Talk"});
      console.log("Try to stop");
      recognition.onend = () => {
        this.props.onSpeech(this.state.finalTranscript);
        console.log("Ended!");
        this.setState({
          finalTranscript: ''
        })
      }
      recognition.stop();
    }

    let finalTranscript = ''
    recognition.onresult = event => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        finalTranscript += transcript + ' ';
      }
      this.setState({finalTranscript: finalTranscript});
    }
  }

  handleCancel() {
    this.setState({
      listening: false,
      finalTranscript: '',
      buttonValue: 'Talk'
    })
    recognition.onend = () => {
      console.log("Ended!")
    }
    recognition.stop();
  }

  render() {
    const { listening, finalTranscript, buttonValue } = this.state;
    return (
      <div style={{display:'flex', flexDirection:'column', alignItems:'center', textAlign:'center'}}>
        <div id='final' style={{color: 'black', border: '#ccc 1px solid', padding: '1em', margin: '1em',
          width: '100%'}}>{finalTranscript}</div>
        <div style={{display:'flex', flexDirection:'row', alignItems:'center', textAlign:'center'}}>
          <button class='button' style={{width:'80px', height:'40px'}}
                  onClick={this.handleClick}>{buttonValue}</button>
          <button class='button' style={{marginLeft:'10px', width:'80px',
            height:'40px'}}
                  onClick={this.handleCancel} hidden={!listening}>Cancel</button>
        </div>
      </div>
    )
  }
}

export default Speech
